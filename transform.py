import torch
import os
from transformers import BertTokenizer, BertModel, BertConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.cuda.set_device(3)

if __name__ == '__main__':
    model_path = '/code/tcw/SFTP/ADB/uncased_L-12_H-768_A-12/'
    save_path = '/code/tcw/SFTP/binary_cls/best_model_banking1.pth'
    fine_tuned_model_dir = "bert"
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"

    tokenizer = BertTokenizer.from_pretrained(model_path)
    Config = BertConfig.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path, config=Config)
    model.load_state_dict(torch.load(save_path), False)

    model_file = os.path.join(fine_tuned_model_dir, WEIGHTS_NAME)
    model_config_file = os.path.join(fine_tuned_model_dir, CONFIG_NAME)
    torch.save(model.state_dict(), model_file)
    model.config.to_json_file(model_config_file)
    tokenizer.save_vocabulary(fine_tuned_model_dir)

    # torch.save(model.state_dict(), 'bert')
    # model.config.to_json_file('bert')
    # tokenizer.save_vocabulary('bert')
    pass
