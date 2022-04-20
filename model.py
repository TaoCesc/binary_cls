import math

from torch.nn import Parameter

from util import *


class BertForModel(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None):
        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                    output_all_encoded_layers=True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        if feature_ext:
            return pooled_output
        elif mode == 'train':
            logits = self.classifier(pooled_output, labels)
            return logits


class SphereFace(nn.Module):
    """ reference: <SphereFace2: Binary Classification is All You Need
                    for Deep Face Recognition>
        margin='C' -> SphereFace2-C
        margin='A' -> SphereFace2-A
        marign='M' -> SphereFAce2-M
    """

    def __init__(self, feat_dim, num_class, magn_type='C',
                 alpha=0.7, r=1., m=0.4, t=3., lw=10.):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type

        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        # init weights
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

        # init bias
        z = alpha / ((1. - alpha) * (num_class - 1.))
        if magn_type == 'C':
            ay = r * (2. * 0.5 ** t - 1. - m)
            ai = r * (2. * 0.5 ** t - 1. + m)
        elif magn_type == 'A':
            theta_y = min(math.pi, math.pi / 2. + m)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.) ** t - 1.)
            ai = r * (2. * 0.5 ** t - 1.)
        elif magn_type == 'M':
            theta_y = min(math.pi, m * math.pi / 2.)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.) ** t - 1.)
            ai = r * (2. * 0.5 ** t - 1.)
        else:
            raise NotImplementedError

        temp = (1. - z) ** 2 + 4. * z * math.exp(ay - ai)
        b = (math.log(2. * z) - ai
             - math.log(1. - z + math.sqrt(temp)))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    """
    def forward(self, x):
        # x: N x D
        # w: D x C
        # b: 1 x C
        # y: N x C
        y = torch.matmul(x, self.w) + self.b
        return y
        
    """

    def forward(self, x, y=None, mode=None):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # delta theta with margin
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        # if mode == 'eval':
        #     return cos_theta
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.)
        with torch.no_grad():
            if self.magn_type == 'C':
                g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.
                g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)
            elif self.magn_type == 'A':
                theta_m = torch.acos(cos_theta.clamp(-1 + 1e-5, 1. - 1e-5))
                theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
                theta_m.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(theta_m)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            elif self.magn_type == 'M':
                m_theta = torch.acos(cos_theta.clamp(-1 + 1e-5, 1. - 1e-5))
                m_theta.scatter_(1, y.view(-1, 1), self.m, reduce='multiply')
                m_theta.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(m_theta)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            else:
                raise NotImplementedError
            d_theta = g_cos_theta - cos_theta

        if mode == 'eval':
            return self.r * (cos_theta - self.m) + self.b
        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)
        weight = self.lw * self.num_class / self.r * weight
        loss = F.binary_cross_entropy_with_logits(
            logits, one_hot, weight=weight)

        return loss


class SphereFace2(nn.Module):
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cosine(m*theta)
    """

    def __init__(self, in_features, out_features, lamb=0.7, r=40, m=0.4, t=3, b=0.20):
        super(SphereFace2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lamb = lamb
        self.r = r
        self.m = m
        self.t = t
        self.b = b
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        self.mlambda = [lambda x: 2 * ((x + 1) / 2) ** self.t - 1, ]

    def forward(self, embedding, label=None, mode=None):
        # cosine(theta) & phi(theta)
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.r * (self.mlambda[0](cos_theta) - self.m) + self.b
        cos_m_theta1 = self.r * (self.mlambda[0](cos_theta) + self.m) + self.b

        cos_p_theta = (self.lamb / self.r) * torch.log(1 + torch.exp(-cos_m_theta))
        cos_n_theta = ((1 - self.lamb) / self.r) * torch.log(1 + torch.exp(cos_m_theta1))
        if mode == 'ext':
            return cos_p_theta
        # covert label to one-hot
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        np.argmin
        # calculate output
        loss = (one_hot * cos_p_theta) + (1 - one_hot) * cos_n_theta
        loss = loss.sum(dim=1)
        return loss.mean()
