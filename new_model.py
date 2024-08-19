import torch
import torch.nn as nn
from net.models.SUM import SUM
from net.configs.config_setting import setting_config
import torchvision.models as models
from GEM import GeneralizedMeanPooling as gem
from whitening import WhiteningNormalization as wnormal

class FullyConnectedNetwork(nn.Module):
    def __init__(self, sum_model, input_features, dropout_rate=0.5):
        super(FullyConnectedNetwork, self).__init__()
        self.sum_model = sum_model
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_1 = nn.BatchNorm1d(num_features=512)
        self.batch_2 = nn.BatchNorm1d(num_features=256)
        self.batch_3 = nn.BatchNorm2d(num_features=768)
        self.batch_4 = nn.BatchNorm2d(num_features=768)
        self.resnet50 = models.resnet50(pretrained=True)
        # Remove the fully connected (fc) layer
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-2])
        self.fc1 = nn.Linear(input_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.conv_resnet = nn.Conv2d(in_channels=2048, out_channels=768, kernel_size=1)
        self.conv_total = nn.Conv2d(in_channels=2 * 768, out_channels=768, kernel_size=1)
        self.drop = nn.Dropout(p=0.5)
        self.gem = gem()

    def forward(self, x):
        x_1 = self.sum_model(x)
        x_1 = x_1.permute(0, 3, 1, 2)
        x_2 = self.resnet50(x)
        x_2 = self.conv_resnet(x_2)
        x_2 = self.batch_3(nn.ReLU()(x_2))
        x_2 = self.drop(x_2)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv_total(x)
        x = self.gem(x)
        x = self.batch_4(nn.ReLU()(x_2))
        x = self.drop(x)
        x = self.flatten(x)
        x = self.batch_1(nn.ReLU()(self.fc1(x)))
        x = self.drop(x)
        x = self.batch_2(nn.ReLU()(self.fc2(x)))
        x = self.drop(x)
        return x

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
    for i in range(len(par)):
        par[i].requires_grad = False
    sum_model.load_from()
    sum_model.cuda()
    input_features = 37632  # Adjust based on your model's output shape
    model = FullyConnectedNetwork(sum_model, input_features)
    return model

if __name__ == "__main__":
    model = load_model()
    model.to('cuda')
    data = torch.rand((8, 3, 224, 224))
    data = data.to('cuda')
    print(model.forward(data).shape)