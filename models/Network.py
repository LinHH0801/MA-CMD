import torch
import torch.nn as nn
from torch.nn import functional as F
from thop import profile
from models.pvtv2 import ourpvt_v2_b1

from mamba_ssm import Mamba


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class SA(nn.Module):
    def __init__(self, in_dim):
        super(SA, self).__init__()

        self.chanel_in = in_dim
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.GAP1 = nn.AdaptiveAvgPool2d(8)
        self.GAP2 = nn.AdaptiveAvgPool2d(8)

    def forward(self, x,y):
        m_batchsize, C, height, width = x.size()
        query_c = self.GAP1(self.query(y))
        key_c = self.GAP2(self.key(x))

        query_c = query_c.view(m_batchsize, -1, width//8 * height//8).permute(0, 2, 1)
        key_c = key_c.view(m_batchsize, -1, width//8 * height//8)

        energy_c = torch.bmm(query_c, key_c)
        attention_c = self.softmax(energy_c)

        return attention_c



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MeasureFusion(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(MeasureFusion, self).__init__()

        self.conv = BasicConv2d(in_channel, out_channel, 3, padding=1)
    def forward(self, x1,x2):

        x = self.conv(torch.abs(x1-x2))
        return x



class Backbone(nn.Module):
    def __init__(self, pretrained_path=r'E:\Lp_8\WUSU\models\pvt_v2_b1.pth'):
        super(Backbone, self).__init__()
        self.backbone = ourpvt_v2_b1()  # [64, 128, 320, 512]
        save_model = torch.load(pretrained_path, weights_only=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.Translayer0 = BasicConv2d(64, 128, 3, padding=1)
        self.Translayer1 = BasicConv2d(128, 128, 3, padding=1)
        self.Translayer2 = BasicConv2d(320, 128, 3, padding=1)
        self.Translayer3 = BasicConv2d(512, 128, 3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.conv = self._make_layer(ResBlock, 128 * 4, 256, 1, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x1):

        x1 = self.backbone(x1)
        x1_1 = self.mp1(self.Translayer0(x1[0]))
        x1_2 = self.Translayer1(x1[1])
        x1_3 = self.up1(self.Translayer2(x1[2]))
        x1_4 = self.up2(self.Translayer3(x1[3]))
        x1_4 = self.conv(torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1))

        return x1_4
class MambaBlock_Temporal(nn.Module):
    def __init__(self, hidden_dim, d_state=128, d_conv=4, expand=2, channel_first=True,downsample_ratio=2):
        super().__init__()
        self.channel_first = channel_first
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.downsample_ratio = downsample_ratio
        self.linear1 = nn.Linear(hidden_dim, hidden_dim//4)
        self.linear2 = nn.Linear(hidden_dim//4, hidden_dim)
        self.LN = nn.LayerNorm(512)
        self.norm = nn.LayerNorm(hidden_dim * downsample_ratio ** 2)
        self.reduction = nn.Linear(hidden_dim * downsample_ratio ** 2, hidden_dim)
        self.GAP = nn.AdaptiveAvgPool2d(8)

        self.pool = nn.MaxPool1d(3)
        self.conv = nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1)
        self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x1,x2,x3,a1,a2,a3):


        x1 = self.GAP(x1)
        x2 = self.GAP(x2)
        x3 = self.GAP(x3)

        m_batchsize,channel,width,height = x1.size()
        x1_a = torch.bmm(x1.view(m_batchsize, -1, width * height), a1.permute(0, 2, 1))
        x2_a = torch.bmm(x2.view(m_batchsize, -1, width * height), a2.permute(0, 2, 1))
        x3_a = torch.bmm(x3.view(m_batchsize, -1, width * height), a3.permute(0, 2, 1))
        x1 = x1_a.view(m_batchsize, channel, height, width)
        x2 = x2_a.view(m_batchsize, channel, height, width)
        x3 = x3_a.view(m_batchsize, channel, height, width)

        x1 = x1.flatten(2).permute(0, 2, 1)
        x2 = x2.flatten(2).permute(0, 2, 1)
        x3 = x3.flatten(2).permute(0, 2, 1)

        x = torch.cat([x1,x2,x3],dim=1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.view(m_batchsize, channel, height, width)
        x = self.up(self.conv(x))


        return x

class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        self.backbone = Backbone()  # [64, 128, 320, 512]
        self.SA = SA(256)

        self.MMA = MambaBlock_Temporal(hidden_dim=256, d_state=64)
        self.cov_diff = self._make_layer(ResBlock, 256*2 , 256, 1, stride=1)
        self.cov = self._make_layer(ResBlock, 256 * 2, 256, 1, stride=1)
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x3 = self.backbone(x3)
        a1 = self.SA(x1, x2)
        a2 = self.SA(x2, x3)
        a3 = self.SA(x1, x3)
        diff_12 = torch.abs(x1 - x2)
        diff_23 = torch.abs(x2 - x3)

        diff = self.cov_diff(torch.cat([diff_12, diff_23], dim=1))
        y  = self.MMA(x1,x2,x3,a1,a2,a3)
        y  = self.cov(torch.concat([y,diff],dim=1))+diff

        return y



class CSTMNet(nn.Module):
    def __init__(self, in_channels=3,num_classes=2):
        super(CSTMNet, self).__init__()
        self.Encoder = Encoder()
        self.head1 = self._make_layer(ResBlock, 256, 64, 1, stride=1)
        self.head2 = self._make_layer(ResBlock, 256, 64, 1, stride=1)
        self.classifier1 = nn.Conv2d(64, num_classes+2, kernel_size=3, padding=1)
        self.classifier2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2,x3):
        x_size = x1.size()

        y = self.Encoder(x1, x2, x3)

        change_moments = self.classifier1(self.head1(y))
        change_binary = self.classifier2(self.head2(y))

        return F.interpolate(change_moments, x_size[2:], mode='bilinear'),F.interpolate(change_binary, x_size[2:], mode='bilinear')


if __name__ == '__main__':
    iterations = 100
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    x1 = torch.randn(1, 3, 512, 512).cuda()
    x2 = torch.randn(1, 3, 512, 512).cuda()
    x3 = torch.randn(1, 3, 512, 512).cuda()
    model = CSTMNet().cuda()
    fine_moment_prob = model(x1, x2, x3)

    flops, params = profile(model, inputs=(x1, x2, x3))[0], profile(model, inputs=(x1, x2, x3))[1]
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    # 测速
    times = torch.zeros(iterations)
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(x1, x2, x3)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times[iter] = curr_time


    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

