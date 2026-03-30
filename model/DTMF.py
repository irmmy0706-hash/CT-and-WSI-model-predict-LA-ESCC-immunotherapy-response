import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.Attention import Attention_Gated as Attention
from model.network import DimReduction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        # 定义卷积层，用于计算权重
        self.conv1 = nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 计算平均值和最大值
        avg_pool = F.avg_pool3d(x, kernel_size=x.size()[2:])
        max_pool = F.max_pool3d(x, kernel_size=x.size()[2:])

        # 将平均值和最大值拼接在一起
        pool = torch.cat([avg_pool, max_pool], dim=1)

        # 使用卷积层计算权重
        weights = torch.sigmoid(self.conv1(pool))

        # 使用权重加权输入
        x = x * weights

        return x


class MultiScaleSpatialAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleSpatialAttention3D, self).__init__()

        # 定义三个卷积层，用于计算不同尺度的权重
        self.conv1 = nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        # 计算平均值和最大值
        avg_pool = F.avg_pool3d(x, kernel_size=x.size()[2:])
        max_pool = F.max_pool3d(x, kernel_size=x.size()[2:])

        # 将平均值和最大值拼接在一起
        pool = torch.cat([avg_pool, max_pool], dim=1)

        # 使用三个不同尺度的卷积核计算权重
        weights1 = torch.sigmoid(self.conv1(pool))
        weights2 = torch.sigmoid(self.conv2(pool))
        weights3 = torch.sigmoid(self.conv3(pool))

        # 使用权重加权输入
        x = x * (weights1 + weights2 + weights3)

        return x


class AdaptiveScaleSpatialAttention3D(nn.Module):
    def __init__(self, in_channels, num_scales=4):
        super(AdaptiveScaleSpatialAttention3D, self).__init__()

        self.num_scales = num_scales

        # 定义多个卷积层，每个卷积层对应不同尺度
        self.convs = nn.ModuleList(
            [nn.Conv3d(in_channels, 1, kernel_size=i, stride=1, padding=i // 2) for i in range(1, num_scales * 2, 2)])

        # 定义自适应权重
        self.adaptive_weights = nn.Parameter(torch.ones(len(self.convs)))

    def forward(self, x):
        scale_weights = F.softmax(self.adaptive_weights, dim=0)

        # 计算每个尺度上的注意力
        attentions = [torch.sigmoid(conv(x)) for conv in self.convs]

        # 加权融合注意力
        weighted_attention = sum(att * weight for att, weight in zip(attentions, scale_weights))

        # 使用权重加权输入
        x = x * weighted_attention

        return x


# 使用预训练的 ResNet-18 模型作为编码器
class ResNet3DEncoder(nn.Module):
    def __init__(self):
        super(ResNet3DEncoder, self).__init__()
        # 使用 torchvision 中的 resnet18 预训练模型
        self.resnet = models.video.r3d_18(pretrained=True)  # 或其他版本的 ResNet-18
        # self.resnet = models.video.mc3_18(pretrained=True)  # 或其他版本的 ResNet-18
        # self.resnet = models.video.r2plus1d_18(pretrained=True)  # 或其他版本的 ResNet-18

    def forward(self, x):
        x = self.resnet.stem(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


class SAttention3D(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SAttention3D, self).__init__()
        # 使用3D卷积
        self.fc1 = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        # 使用3D自适应平均池化
        x1 = F.adaptive_avg_pool3d(inputs, output_size=(1, 1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        # 使用3D自适应最大池化
        x2 = F.adaptive_max_pool3d(inputs, output_size=(1, 1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        # 合并两个池化路径的结果
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1, 1)
        return x


class ModelImages(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ModelImages, self).__init__()

        self.encoder = ResNet3DEncoder()
        self.bn = nn.BatchNorm3d(num_features=512)  # 使用BN层，注意输入通道数要匹配
        self.fc = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.attention = MultiScaleSpatialAttention3D(in_channels=6)
        self.ca = SAttention3D(input_channels=3, internal_neurons=3)

    def forward(self, x):
        x = self.attention(x)
        channel_att_vec = self.ca(x)
        x = channel_att_vec * x
        x = self.encoder(x)
        x = self.bn(x)
        self.feature_maps = x  # 保存特征图以便于后续生成CAM
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_activation_maps(self):
        return self.feature_maps

# 合并模型
class MergeModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_images = ModelImages()
        self.fc = nn.Linear(512, num_classes)  # yuanshi
        self.attention = Attention(512)
        self.dimReduction = DimReduction(1024, 512, numLayer_Res=0)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, images1, pathology):
        images1_output = self.model_images(images1)  # (B, 512)
        final_feats = []
        for i in range(len(pathology)):
            img_feat = images1_output[i]  # (1,512)
            p_feat = pathology[i]  # (n_i, 1024) 或 None
            if p_feat is None:
                final_feats.append(img_feat.unsqueeze(0))
            else:
                p_feat_reduced = self.dimReduction(p_feat)  # (n_i, 512)
                p_feat_mean = torch.mean(p_feat_reduced, dim=0)  # (1,512)
                feat = torch.stack([img_feat, p_feat_mean], dim=0)  # (2, 512)
                AA = self.attention(feat)  # (1, 2)
                afeat = torch.mm(AA, feat)  # (1, 512)
                final_feats.append(afeat)

        # 把所有样本的特征拼起来，变成 (B, 512)
        final_feats_tensor = torch.stack(final_feats, dim=0)

        x = F.relu(self.fc(final_feats_tensor)).squeeze(1)
        return x, images1_output

    def get_activation_maps(self):
        return self.model_images.get_activation_maps(), self.fc.weight.data

# def main():
#     # 配置参数
#     batch_size = 10
#     in_channels = 3  # 或 1，根据你的输入设置
#     depth = 10       # 切片数，对应 3D 中的 "时间/深度" 维度
#     height = 64
#     width = 64
#
#     # 构造一个模拟输入 (B, C, D, H, W)
#     dummy_input = torch.randn(batch_size, in_channels, depth, height, width)
#     pathology = create_random_pathology_tensor_list(batch_size)
#     # 初始化模型
#     model = MergeModel(num_classes=2)
#     print(f"Input CT shape: {dummy_input.shape}")
#
#     # 前向传播
#     output,_ = model(dummy_input,pathology)
#     print(output.shape)
# if __name__ == '__main__':
#     main()