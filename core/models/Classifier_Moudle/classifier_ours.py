import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from core.models.Classifier_Moudle.FFCmodules import ffc

torch.manual_seed(1234)


class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, stride):
        super(ASPPBlock, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=padding, dilation=dilation, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.relu(self.bn(x))
        return x


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBottleneck, self).__init__()
        # in_channels = 2048
        dilations = [6, 12, 18, 24]

        self.aspp1 = ASPPBlock(in_channels, 16, kernel_size=1, dilation=dilations[0], padding=0, stride=1)
        self.aspp2 = ASPPBlock(in_channels, 16, kernel_size=3, dilation=dilations[1], padding=dilations[1], stride=1)
        self.aspp3 = ASPPBlock(in_channels, 16, kernel_size=3, dilation=dilations[2], padding=dilations[2], stride=1)
        self.aspp4 = ASPPBlock(in_channels, 16, kernel_size=3, dilation=dilations[3], padding=dilations[3], stride=1)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                             nn.Conv2d(in_channels, 16, kernel_size=1, stride=1),
                                             nn.BatchNorm2d(16),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(80, 16, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x_output = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x_output = self.conv1(x_output)
        x_output = self.relu(self.bn1(x_output))
        x_output = self.conv2(x_output)
        return x_output


class ChannelGrouping(nn.Module):
    def __init__(self, in_channels, num_local_channels=1024, num_global_channels=1024):
        super(ChannelGrouping, self).__init__()
        self.local_channels = num_local_channels
        self.global_channels = num_global_channels

        self.local_conv = nn.Conv2d(in_channels=in_channels, out_channels=num_local_channels, kernel_size=1)
        self.global_conv = nn.Conv2d(in_channels=in_channels, out_channels=num_global_channels, kernel_size=1)

    def forward(self, x):
        local_feature = self.local_conv(x)
        global_feature = self.global_conv(x)
        return local_feature, global_feature


class Classifier(nn.Module):
    def __init__(self, n_classes=6):
        super(Classifier, self).__init__()
        self.channel_group = ChannelGrouping(2048)
        self.ffc = ffc.SJTNet_FFCBlock(2048)
        self.ffm = ffc.FeatureFusionMoudle(2048, 2048)      # [b, 2048, 64, 64]
        self.convbottle = ConvBottleneck(2048, n_classes)    # [b, n_classes, 64, 64]

    def forward(self, x):
        x_local, x_global = self.channel_group(x)
        x5_local, x5_global = self.ffc(x_local, x_global)
        x_ffm = self.ffm(x5_local, x5_global)
        x_output = self.convbottle(x_ffm)
        return x_output


if __name__ == "__main__":
    model = Classifier(n_classes=6)
    model.cuda()
    image = torch.randn(2, 2048, 64, 64).cuda()
    print(model)
    # print(model(image)[0].shape)
    # print(model(image)[1].shape)
    print(model(image).shape)         #  output_shape = [2, 6, 64, 64]