import torch
import torch.nn as nn
from net.models.SUM import SUM
from net.configs.config_setting import setting_config

class FullyConnectedNetwork(nn.Module):
    def __init__(self, sum_model, input_features, dropout_rate=0.5):
        super(FullyConnectedNetwork, self).__init__()
        self.sum_model = sum_model
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn2d = nn.BatchNorm2d(num_features=24)
        #self.fc1_1 = nn.Linear(input_features, 256)
        self.conv = nn.Conv2d(in_channels=768, out_channels=24, kernel_size=1)

    def forward(self, x):
        x = self.sum_model(x)
        x = x.permute(0, 3, 1, 2)
        out = self.bn2d(self.conv(x))
        out = self.flatten(out)
        return out

def load_model():
    config = setting_config
    model_cfg = config.model_config
    sum_model = SUM(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    par = list(sum_model.parameters())
    # for i in range(int(len(par) / 1.5)):
    #     par[i].requires_grad = False
    sum_model.load_from()
    sum_model.cuda()
    input_features = 37632  # Adjust based on your model's output shape
    model = FullyConnectedNetwork(sum_model, input_features)
    return model