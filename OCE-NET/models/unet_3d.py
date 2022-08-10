import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    pad = int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)
    return pad

class LesionGateConv(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), use_gate=False):
        super(LesionGateConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm3d = torch.nn.BatchNorm3d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.use_gate = use_gate

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv3d(input)
        mask = self.mask_conv3d(input)

        # gated features
        if self.batch_norm:
            x = self.batch_norm3d(x)

        if self.activation is not None:
            if self.use_gate:
                x = self.activation(x) * self.gated(mask)
            else:
                x = self.activation(x)
        else:
            if self.use_gate:
                x = x * self.gated(mask)
        return x


class LesionGateDeConv(torch.nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(LesionGateDeConv, self).__init__()
        # self.conv3d = LesionGateConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation, use_gate=False)
        # self.scale_factor = scale_factor

        self.conv3d2 = nn.Sequential(
                        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm3d(out_channels),
                        activation)

    def forward(self, input):
        # x = F.interpolate(input, scale_factor=self.scale_factor)
        # x = self.conv3d(x)

        x = self.conv3d2(input)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_dim=45, out_dim=45, batch_norm=True, cnum=32):
        super(UNet3D, self).__init__()
        print("UNet3D, cmun = 32")
        self.in_dim = in_dim
        self.out_dim = out_dim
        activation = nn.ReLU(inplace=True)
        # activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.enc1_1 = LesionGateConv(in_dim, cnum, 3, 1, padding=get_pad(64, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        self.enc1_2 = LesionGateConv(cnum, cnum, 3, 2, padding=get_pad(64, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        # downsample to 128
        self.enc2_1 = LesionGateConv(cnum, 2 * cnum, 3, 1, padding=get_pad(32, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        self.enc2_2 = LesionGateConv(2 * cnum, 2 * cnum, 3, 2, padding=get_pad(32, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        # downsample to 64
        self.enc3_1 = LesionGateConv(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(16, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        self.enc3_2 = LesionGateConv(4 * cnum, 4 * cnum, 3, 2, padding=get_pad(16, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        # downsample to 32
        self.enc4_1 = LesionGateConv(4 * cnum, 8 * cnum, 3, 1, padding=get_pad(8, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        self.enc4_2 = LesionGateConv(8 * cnum, 8 * cnum, 3, 2, padding=get_pad(8, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)

        # Bridge
        self.bridge = LesionGateConv(8 * cnum, 16 * cnum, 3, 1, padding=get_pad(4, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)

        # Up sampling
        self.dec1_1 = LesionGateDeConv(2, 16 * cnum, 8 * cnum, 3, 1, padding=get_pad(8, 3, 1), batch_norm=batch_norm, activation=activation)
        self.dec1_2 = LesionGateConv(16 * cnum, 8 * cnum, 3, 1, padding=get_pad(8, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        self.dec2_1 = LesionGateDeConv(2, 8 * cnum, 4 * cnum, 3, 1, padding=get_pad(16, 3, 1), batch_norm=batch_norm, activation=activation)
        self.dec2_2 = LesionGateConv(8 * cnum, 4 * cnum, 3, 1, padding=get_pad(16, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        self.dec3_1 = LesionGateDeConv(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(32, 3, 1), batch_norm=batch_norm, activation=activation)
        self.dec3_2 = LesionGateConv(4 * cnum, 2 * cnum, 3, 1, padding=get_pad(32, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)
        self.dec4_1 = LesionGateDeConv(2, 2 * cnum, cnum, 3, 1, padding=get_pad(64, 3, 1), batch_norm=batch_norm, activation=activation)
        self.dec4_2 = LesionGateConv(2 * cnum, cnum, 3, 1, padding=get_pad(64, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=False)

        # Output
        self.out = LesionGateConv(cnum, out_dim, 3, 1, padding=get_pad(64, 3, 1), batch_norm=batch_norm, activation=None, use_gate=False)

    def forward(self, x, encoder_only=False, save_feat=False, lgc_layers=['enc4_1', 'enc3_1', 'enc2_1']):
        feat = []

        # x: b c w h d
        # Down sampling
        down_1 = self.enc1_1(x)
        pool_1 = self.enc1_2(down_1)

        down_2 = self.enc2_1(pool_1)
        pool_2 = self.enc2_2(down_2)
        x = pool_2
        if 'enc2_1' in lgc_layers:
            feat.append(x)

        down_3 = self.enc3_1(pool_2)
        pool_3 = self.enc3_2(down_3)
        x = pool_3
        if 'enc3_1' in lgc_layers:
            feat.append(x)

        down_4 = self.enc4_1(pool_3)
        pool_4 = self.enc4_2(down_4)
        x = pool_4
        if 'enc4_1' in lgc_layers:
            feat.append(x)
        # print('pool shape', pool_1.shape, pool_2.shape, pool_3.shape, pool_4.shape)

        if encoder_only:
            return feat

        # Bridge
        bridge = self.bridge(pool_4)

        # Up sampling
        trans_1 = self.dec1_1(bridge)
        # print(trans_1.shape, down_4.shape)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.dec1_2(concat_1)

        trans_2 = self.dec2_1(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.dec2_2(concat_2)

        trans_3 = self.dec3_1(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.dec3_2(concat_3)

        trans_4 = self.dec4_1(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.dec4_2(concat_4)
        # print('up shape', up_1.shape, up_2.shape, up_3.shape, up_4.shape)

        # Output
        out = self.out(up_4)

        if save_feat:
            return out, feat
        else:
            return out
