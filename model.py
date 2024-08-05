import torch
import torch.nn as nn
from net.models.SUM import SUM
from net.configs.config_setting import setting_config

class FullyConnectedNetwork(nn.Module):
    def __init__(self, sum_model, num_classes1, num_classes2, num_classes3, input_features, dropout_rate=0.5):
        super(FullyConnectedNetwork, self).__init__()
        self.sum_model = sum_model
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1_1 = nn.Linear(input_features, 512)
        self.bn1_1 = nn.BatchNorm1d(512)
        self.fc2_1 = nn.Linear(512, 256)
        self.bn2_1 = nn.BatchNorm1d(256)
        self.fc3_1 = nn.Linear(256, num_classes1)
        self.fc1_2 = nn.Linear(input_features, 256)
        self.bn1_2 = nn.BatchNorm1d(256)
        self.fc2_2 = nn.Linear(256, 128)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.fc3_2 = nn.Linear(128, num_classes2)
        self.fc1_3 = nn.Linear(input_features, 128)
        self.bn1_3 = nn.BatchNorm1d(128)
        self.fc2_3 = nn.Linear(128, 64)
        self.bn2_3 = nn.BatchNorm1d(64)
        self.fc3_3 = nn.Linear(64, num_classes3)

    def forward(self, x):
        x = self.sum_model(x)
        x = self.flatten(x)
        out1 = torch.relu(self.bn1_1(self.fc1_1(x)))
        out1 = self.dropout(out1)
        out1 = torch.relu(self.bn2_1(self.fc2_1(out1)))
        out1 = self.dropout(out1)
        out1 = self.fc3_1(out1)
        out2 = torch.relu(self.bn1_2(self.fc1_2(x)))
        out2 = self.dropout(out2)
        out2 = torch.relu(self.bn2_2(self.fc2_2(out2)))
        out2 = self.dropout(out2)
        out2 = self.fc3_2(out2)
        out3 = torch.relu(self.bn1_3(self.fc1_3(x)))
        out3 = self.dropout(out3)
        out3 = torch.relu(self.bn2_3(self.fc2_3(out3)))
        out3 = self.dropout(out3)
        out3 = self.fc3_3(out3)
        return out1, out2, out3

def load_model(num_classes1, num_classes2, num_classes3):
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
    for param in sum_model.parameters():
        param.requires_grad = False
    sum_model.load_from()
    sum_model.cuda()
    input_features = 37632  # Adjust based on your model's output shape
    model = FullyConnectedNetwork(sum_model, num_classes1, num_classes2, num_classes3, input_features)
    return model
