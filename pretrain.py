import torch.optim
from sklearn import manifold
from sklearn.neighbors import LocalOutlierFactor
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from transformers import get_linear_schedule_with_warmup

from init_parameter import init_model

from model import *
from dataloader import *
from util import *


class PretrainModelManager:

    def __init__(self, args, data):

        self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="", num_labels=data.num_labels)
        for name, param in self.model.named_parameters():
            param.requires_grad = True

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.num_train_optimization_steps = int(
            len(data.train_examples) / args.train_batch_size) * args.num_train_epochs
        self.sphereface = SphereFace(feat_dim=768, num_class=data.num_labels, magn_type='C',
                                     # alpha=round((data.num_labels - 1.0) / (data.num_labels - 0.0), 4)
                                     ).to(self.device)

        self.optimizer_bert, self.optimizer_mlp, self.scheduler = self.get_optimizer(args)
        self.best_eval_score = 0
        self.best_loss = 500

    def eval(self, args, data):

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                cos_theta = self.sphereface(features, y=label_ids, mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, cos_theta))

        y_pred = np.argmax(total_logits.cpu().detach().numpy(), axis=1)
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc

    def test(self, args, data):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty(0, dtype=torch.long).to(self.device)
        for batch in tqdm(data.test_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                cos_theta = self.sphereface(features, y=label_ids, mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, cos_theta))

        y_pred = np.argmax(total_logits.cpu().detach().numpy(), axis=1)
        y_true = total_labels.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        results = F_measure(cm)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results["Accuracy"] = acc

        self.test_results = results
        self.save_results(args)

    def train(self, args, data):

        wait = 0
        best_model = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss = self.sphereface(features, label_ids)
                    self.optimizer_bert.zero_grad()
                    self.optimizer_mlp.zero_grad()
                    loss.backward()
                    self.optimizer_bert.step()
                    self.optimizer_mlp.step()
                    self.scheduler.step()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print("Epoch: {}/{}".format(epoch, int(args.num_train_epochs)))
            print("Loss: {}".format(loss))
            acc = self.eval(args, data)
            print("Accuracy: {}".format(acc))
            if epoch == 5:
                self.eval(args, data)
            if acc > self.best_eval_score:
                self.best_eval_score = acc
                self.best_loss = loss
                wait = 0
                best_model = copy.deepcopy(self.model)
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        self.model = best_model

    def get_optimizer(self, args):

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer_bert = BertAdam(optimizer_grouped_parameters,
                                  lr=args.lr,
                                  warmup=args.warmup_proportion,
                                  t_total=self.num_train_optimization_steps)

        optimizer_mlp = torch.optim.SGD(self.sphereface.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_mlp, milestones=[1000, 2000, 3000], gamma=0.1)
        return optimizer_bert, optimizer_mlp, scheduler

    def get_optimizer1(self, args):
        optimizer = SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        scheduler = MultiStepLR(optimizer, milestones=[40000, 60000, 70000], gamma=0.1)
        return optimizer, scheduler

    def save_model(self, args):

        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model

        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        # np.save(os.path.join(args.save_results_path, 'centroids.npy'), self.centroids.detach().cpu().numpy())
        # np.save(os.path.join(args.save_results_path, 'deltas.npy'), self.delta.detach().cpu().numpy())

        file_name = 'binary_fine-tune.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)

    def t_SNE(self, data, labels):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(data.cpu().detach().numpy())
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)
        plt.figure(figsize=(8, 8))
        # labels = labels.cpu().detach().numpy()
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def get_outputs(self, args, data, mode, get_feats=False, train_feats=None):

        if mode == 'train':
            dataloader = self.train_dataloader
        elif mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_dim)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(input_ids, segment_ids, input_mask)

                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, pooled_output))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats
        else:
            total_probs, y_pred = total_logits.max(dim=1)
            y_pred = y_pred.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            if train_feats is not None:
                feats = total_features.cpu().numpy()
                y_pred = self.classify_lof(args, data, y_pred, train_feats, feats)

            return y_true, y_pred

    def classify_lof(self, args, data, preds, train_feats, pred_feats):

        lof = LocalOutlierFactor(n_neighbors=args.n_neighbors, contamination=args.contamination, novelty=True,
                                 n_jobs=-1)
        lof.fit(train_feats)
        y_pred_lof = pd.Series(lof.predict(pred_feats))
        preds[y_pred_lof[y_pred_lof == -1].index] = data.unseen_label_id

        return preds


if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)
    print('Pre-training begin...')
    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    print('Pre-training finished!')
    manager_p.test(args, data)
