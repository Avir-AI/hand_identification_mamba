import torch
import torch.nn as nn
from net.models.SUM import SUM
from net.configs.config_setting import setting_config
import torch.nn.functional as F

##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class FullyConnectedNetwork(nn.Module):
    def __init__(self, sum_model, num_classes1, num_classes2, num_classes3, input_features, dropout_rate=0.5):
        super(FullyConnectedNetwork, self).__init__()
        self.sum_model = sum_model
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Prompt blocks for each head
        self.prompt_block1 = PromptGenBlock(prompt_dim=768, prompt_len=5, prompt_size=8, lin_dim=768)
        self.prompt_block2 = PromptGenBlock(prompt_dim=768, prompt_len=5, prompt_size=8, lin_dim=768)
        self.prompt_block3 = PromptGenBlock(prompt_dim=768, prompt_len=5, prompt_size=8, lin_dim=768)
        self.adjust_conv_layer = nn.Conv2d(768 * 2, 768, kernel_size=1)
        
        # first fully connected layers
        self.fc1_1 = nn.Linear(input_features, 512)
        self.bn1_1 = nn.BatchNorm1d(512)
        self.fc2_1 = nn.Linear(512, 256)
        self.bn2_1 = nn.BatchNorm1d(256)
        self.fc3_1 = nn.Linear(256, num_classes1)
        # second fully connected layers
        self.fc1_2 = nn.Linear(input_features, 256)
        self.bn1_2 = nn.BatchNorm1d(256)
        self.fc2_2 = nn.Linear(256, 128)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.fc3_2 = nn.Linear(128, num_classes2)
        # third fully connected layers
        self.fc1_3 = nn.Linear(input_features, 128)
        self.bn1_3 = nn.BatchNorm1d(128)
        self.fc2_3 = nn.Linear(128, 64)
        self.bn2_3 = nn.BatchNorm1d(64)
        self.fc3_3 = nn.Linear(64, num_classes3)
        
    def forward(self, x):
        x = self.sum_model(x) # torch.Size([B, 7, 7, 768])
        
        ### APPLY Prompt for first head
        prompt_input1 = x.permute(0, 3, 1, 2)  # torch.Size([B ,768, 7, 7])
        prompt_output1 = self.prompt_block1(prompt_input1) # torch.Size([B ,768, 7, 7])
        prompt_output1 = torch.cat([prompt_input1, prompt_output1], dim=1) # torch.Size([B, 1536, 7, 7])
        prompt_output1 = self.adjust_conv_layer(prompt_output1) # torch.Size([B, 768, 7, 7])
        x1 = prompt_output1.permute(0, 2, 3, 1) # torch.Size([B, 7, 7, 768])
        x1 = self.flatten(x1)
        
        ### APPLY Prompt for second head
        prompt_input2 = x.permute(0, 3, 1, 2)  # torch.Size([B ,768, 7, 7])
        prompt_output2 = self.prompt_block2(prompt_input2) # torch.Size([B ,768, 7, 7])
        prompt_output2 = torch.cat([prompt_input2, prompt_output2], dim=1) # torch.Size([B, 1536, 7, 7])
        prompt_output2 = self.adjust_conv_layer(prompt_output2) # torch.Size([B, 768, 7, 7])
        x2 = prompt_output2.permute(0, 2, 3, 1) # torch.Size([B, 7, 7, 768])
        x2 = self.flatten(x2)
        
        ### APPLY Prompt for third head
        prompt_input3 = x.permute(0, 3, 1, 2)  # torch.Size([B ,768, 7, 7])
        prompt_output3 = self.prompt_block3(prompt_input3) # torch.Size([B ,768, 7, 7])
        prompt_output3 = torch.cat([prompt_input3, prompt_output3], dim=1) # torch.Size([B, 1536, 7, 7])
        prompt_output3 = self.adjust_conv_layer(prompt_output3) # torch.Size([B, 768, 7, 7])
        x3 = prompt_output3.permute(0, 2, 3, 1) # torch.Size([B, 7, 7, 768])
        x3 = self.flatten(x3)
        
        # Forward through the first separate fully connected network
        out1 = torch.relu(self.bn1_1(self.fc1_1(x1)))
        out1 = self.dropout(out1)
        out1 = torch.relu(self.bn2_1(self.fc2_1(out1)))
        out1 = self.dropout(out1)
        out1 = self.fc3_1(out1)
        
        # Forward through the second separate fully connected network
        out2 = torch.relu(self.bn1_2(self.fc1_2(x2)))
        out2 = self.dropout(out2)
        out2 = torch.relu(self.bn2_2(self.fc2_2(out2)))
        out2 = self.dropout(out2)
        out2 = self.fc3_2(out2)
        
        # Forward through the third separate fully connected network
        out3 = torch.relu(self.bn1_3(self.fc1_3(x3)))
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
    # for param in sum_model.parameters():
    #     param.requires_grad = False
    sum_model.load_from()
    sum_model.cuda()
    input_features = 7 * 7 * 768  # Adjust this value based on your model's output shape
    model = FullyConnectedNetwork(sum_model, num_classes1, num_classes2, num_classes3, input_features)
    return model