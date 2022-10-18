import argparse
import time
import torch.backends.cudnn as cudnn
import torch
from osgeo import gdal
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from module3d import ConvAttention, PreNorm, FeedForward, SepConv3d, MultiHeadAttention
import numpy as np
import scipy.io as sio
import torch.utils.data as Data
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
相比与cvt模型，将卷积映射换成conv3d增强对光谱信息的捕捉
'''


# class Transformer(nn.Module):
#     def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
#             ]))

#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0, groups=1):
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU6(inplace=True)
        )


class Transformer(nn.Module):
    # 去掉mlp_dim参数
    def __init__(self, dim, img_size, depth, heads, dim_head, channel_depth=1, dropout=0., last_stage=False,
                 mobile_block=False, mode="ViT"):
        super().__init__()
        self.layers = nn.ModuleList([])
        mlp_dim = dim * 4
        # if mobile_block:

        #     img_size=int(math.ceil(img_size / 2))
        #     channel_depth=int(math.ceil(channel_depth/2))
        #     num_patches = img_size * img_size * channel_depth
        # else:
        #     num_patches = img_size * img_size * channel_depth
        for _ in range(depth):
            if last_stage == True:
                self.layers.append(nn.ModuleList([

                    PreNorm(dim, ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout,
                                               last_stage=last_stage)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
            else:
                self.layers.append(nn.ModuleList([

                    PreNorm(dim, MultiHeadAttention(dim, heads)),
                    PreNorm(dim, FeedForward(dim, dim * 2, dropout=dropout, act_layer=SiLU()))
                ]))
        # num_patches=img_size*img_size*channels_depth
        self.mode = mode
        # self.skipcat = nn.ModuleList([])
        # for _ in range(depth - 2):
        #     if last_stage==False:
        #         self.skipcat.append(nn.Conv2d(num_patches , num_patches , [1, 2], 1, 0))
        #     else:
        #         self.skipcat.append(nn.Conv2d(num_patches+1, num_patches+1, [1, 2], 1, 0))

    def forward(self, x):
        # for attn, ff in self.layers:
        #     x = attn(x) + x
        #     x = ff(x) + x

        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](

                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x) + x
                x = ff(x) + x
                nl += 1
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, padding, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, padding=padding, groups=hidden_channel),
            # 1x1 pointwise conv(linear)=padding
            nn.Conv3d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


from typing import Optional, Tuple, Union, Dict
import math
from torch import Tensor
from torch.nn import functional as F


class SiLU(torch.nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Optional[Union[int, Tuple[int, int, int]]] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = False,
            use_norm: Optional[bool] = True,
            use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm3d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,
            img_size: int,
            channel_depth: int = 1,
            # ffn_dim: int,
            depth: int = 2,
            num_heads: int = 4,
            attn_dropout: float = 0.0,
            dropout: float = 0.1,
            ffn_dropout: float = 0.0,
            patch_h: int = 2,
            patch_w: int = 2,
            patch_d: int = 2,
            conv_ksize: Optional[int] = 3,
            last_stage: bool = False,
            mobile_block: bool = True,
            mode: str = "ViT",
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        # assert transformer_dim % head_dim == 0
        # num_heads = transformer_dim // hea1d_dim

        global_rep = [Transformer(
            # dim ffn_latent_dim:mlp_dim
            dim=transformer_dim,
            img_size=img_size,
            channel_depth=channel_depth,
            dim_head=64,
            depth=depth,
            heads=num_heads,
            # attn_dropout=attn_dropout,
            dropout=dropout,
            # ffn_dropout=ffn_dropout
            last_stage=last_stage,
            mobile_block=mobile_block,
            mode=mode,
        )]

        # for _ in range(n_transformer_blocks)

        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_d = patch_d
        self.patch_area = self.patch_w * self.patch_h * self.patch_d

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        # self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = depth
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h, patch_d = self.patch_w, self.patch_h, self.patch_d
        patch_area = patch_w * patch_h * patch_d
        # print("----------------x={}---------------".format(x.shape))
        batch_size, in_channels, orig_h, orig_w, orig_d = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
        new_d = int(math.ceil(orig_d / self.patch_d) * self.patch_d)

        interpolate = False
        if new_w != orig_w or new_h != orig_h or new_d != orig_d:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w, new_d), mode="trilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patch_d = new_d // patch_d  # n_d
        num_patches = num_patch_h * num_patch_w * num_patch_d  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        # x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        # x = x.transpose(1, 2)
        # # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        #
        # x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # # [B, C, N, P] -> [B, P, N, C]
        # x = x.transpose(1, 3)
        # # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w, orig_d),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
            "num_patches_d": num_patch_d,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]
        num_patch_d = info_dict["num_patches_d"]

        # [B, P, N, C] -> [B, C, N, P]
        # x = x.transpose(1, 3)
        # # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        # x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        # x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w,
                      num_patch_d * self.patch_d)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="trilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class CvT(nn.Module):
    def __init__(self, image_size, channels, num_classes, dim=64, kernels=[3, 3, 3], strides=[2, 2, 2],
                 heads=[2, 4, 4], depth=[2, 4, 6], pool='cls', dropout=0.1, emb_dropout=0.1, scale_dim=4, mode="ViT"):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        self.heads = heads
        ##### Stage 1 #######
        # n为光谱维度
        # self.conv1 = InvertedResidual(1, dim,2,1, 6)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=dim, kernel_size=kernels[0], stride=strides[0], padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
        )
        # self.stage1_conv_embed = nn.Sequential(
        #     nn.Conv3d(1, dim, kernels[0], strides[0],1),
        #     # Rearrange('b c h w n-> b (h w n) c', h=image_size // 2, w=image_size // 2),
        #     # nn.LayerNorm(dim)
        # )
        self.stage1_transformer = nn.Sequential(
            MobileViTBlock(in_channels=dim, transformer_dim=96, img_size=(image_size + 1) // 2,
                           channel_depth=(channels + 1) // 2,
                           depth=depth[0], num_heads=heads[0], mode=mode),
            # Transformer(dim=dim, img_size= (image_size+1)//2,depth=depth[0], heads=heads[0], dim_head=self.dim,
            #                                   mlp_dim=dim * scale_dim, num_patches=((image_size+1)//2)*((image_size+1)//2),dropout=dropout,mode=mode),

            # Rearrange('b (h w) c -> b c h w', h = image_size//2, w =  image_size//2)
        )

        ##### Stage 2 #######
        in_channels = dim
        scale = 2
        channels = (channels + 1) // 2
        image_size = (image_size + 1) // 2
        dim = scale * dim
        # self.conv2 = InvertedResidual(in_channels, dim,2,1, 6)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, dim, kernels[1], strides[1], 1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
        )
        # self.stage2_conv_embed = nn.Sequential(
        #     # nn.Conv3d(in_channels, dim, kernels[1], strides[1],1),
        #     Rearrange('b c h w n-> b (h w n) c', h=(image_size + 1) // 2, w=(image_size + 1) // 2),
        #     nn.LayerNorm(dim)
        # )
        self.stage2_transformer = nn.Sequential(
            MobileViTBlock(in_channels=dim, transformer_dim=120, img_size=(image_size + 1) // 2,
                           channel_depth=(channels + 1) // 2, depth=depth[1], num_heads=heads[1], mode=mode),
            # Transformer(dim=dim, img_size=(image_size + 1) // 2, depth=depth[1], heads=heads[1],channel_depth=(channels + 1) // 2, dim_head=self.dim,
            #              dropout=dropout,mode=mode),
            # Rearrange('b (h w n) c -> b c h w n', h=(image_size + 1) // 2, w=(image_size + 1) // 2)
        )

        ##### Stage 3 #######
        in_channels = dim
        # //((heads[1]//heads[0]))
        scale = 2
        channels = (channels + 1) // 2
        image_size = (image_size + 1) // 2
        dim = (scale * dim)
        self.conv3 = InvertedResidual(in_channels, dim, 2, 1, 6)
        # self.conv3 = nn.Sequential(
        #     nn.Conv3d(in_channels, dim, kernels[2], strides[2], 1),
        #     nn.BatchNorm3d(dim),
        #     nn.ReLU(inplace=True),
        # )
        self.stage3_conv_embed = nn.Sequential(
            # nn.Conv3d(in_channels, dim, kernels[2], strides[2],1),
            Rearrange('b c h w n-> b (h w n) c', h=(image_size + 1) // 2, w=(image_size + 1) // 2),
            nn.LayerNorm(dim)
        )
        self.stage3_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=(image_size + 1) // 2, depth=depth[2],
                        heads=heads[2], channel_depth=(channels + 1) // 2, dim_head=self.dim,
                        dropout=dropout, last_stage=True, mode=mode),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        conv1 = self.conv1(img)
        xs_trans1 = self.stage1_transformer(conv1)

        conv2 = self.conv2(xs_trans1)
        # xs_conv2 = self.stage2_conv_embed(conv2)
        xs_trans2 = self.stage2_transformer(conv2)

        conv3 = self.conv3(xs_trans2)
        xs_conv3 = self.stage3_conv_embed(conv3)
        b, n, _ = xs_conv3.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        xs_cat = torch.cat((cls_tokens, xs_conv3), dim=1)
        xs_drop = self.dropout_large(xs_cat)
        xs_trans3 = self.stage3_transformer(xs_drop)
        xs = xs_trans3.mean(dim=1) if self.pool == 'mean' else xs_trans3[:, 0]

        xs = self.mlp_head(xs)
        return xs


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    # maxk = max((1,))  # 取top1准确率，若取top1和top5准确率改为max((1,5))
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # 类似于resize
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    train_loader = tqdm(train_loader, file=sys.stdout)
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()  # (32,200,25)
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred = model(batch_data)  # (32,16)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


# def output_metric(tar, pre):
#     matrix = confusion_matrix(tar, pre)
#     OA, AA_mean, Kappa, AA = cal_results(matrix)
#     return OA, AA_mean, Kappa, AA
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    classification = classification_report(tar, pre, digits=4)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA, matrix, classification


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    valid_loader = tqdm(valid_loader, file=sys.stdout)
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


# 边界拓展：镜像
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # 中心区域
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


def chooose_train_and_test_point(train_data, num_classes):
    number_train = []
    pos_train = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        # 为了找出标签值的索引位置(排除无关值的干扰)
        each_class = np.argwhere(train_data == i + 1)
        class1 = each_class[:(each_class.shape[0]), :]
        number_train.append(class1.shape[0])
        # pos_train为1到16的拥有标签值的字典
        pos_train[i] = class1
    # 标签值为1的样本
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        # np.r_()增加行数，在列的方向堆叠，列数不变，行数增加
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    # 将所有的标签值的样本值叠加
    total_pos_train = total_pos_train.astype(int)

    return total_pos_train, number_train


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    # 中心区域
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    # 左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    # 右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[
                                                                                                        :, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[
                                                                                                        :, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    return x_train_band


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, patch=5):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    with tqdm(total=train_point.shape[0], file=sys.stdout) as pbar:
        for i in range(train_point.shape[0]):
            pbar.update(1)
            x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    return x_train


def train_and_test_label(number_train, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
    y_train = np.array(y_train)
    return y_train


def padwithzeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def creatCube(X, y, windowsize=25, removeZeroLabels=True):
    margin = int((windowsize - 1) / 2)
    zeroPaddedX = padwithzeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowsize, windowsize, X.shape[2]))
    print('patchesData.shape ', patchesData.shape)
    patchesLabels = np.zeros(X.shape[0] * X.shape[1])
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
        return patchesData, patchesLabels


def pca_change(X, num_components=10):
    print(f'X.shape{X.shape}')
    newX = np.reshape(X, (-1, X.shape[2]))
    print(f'newX.shape{newX.shape}')
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    print(f'newX2.shape{newX.shape}')
    newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    return newX


def divide_train_and_test_point(train_data, num_classes):
    number_train = []
    pos_train = {}
    train1 = []
    t1_lab = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        # 为了找出标签值的索引位置(排除无关值的干扰)
        each_class = np.argwhere(train_data == i + 1)
        # 此处设置需要注意 Honghu 15 Hanchuan 10 Loukou 10
        class1 = each_class[:int(each_class.shape[0] / 21), :]
        train1.append(class1.shape[0])
        # pos_train为1到16的拥有标签值的字典
        # pos_train[i] = each_class
        t1_lab[i] = class1
        # t2_lab[i] = class2
    # 标签值为1的样本
    # total_pos_train = pos_train[0]
    total_t1_lab = t1_lab[0]
    # total_t2_lab = t2_lab[0]
    for i in range(1, num_classes):
        # np.r_()增加行数，在列的方向堆叠，列数不变，行数增加
        # total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
        total_t1_lab = np.r_[total_t1_lab, t1_lab[i]]  # (695,2)
        # total_t2_lab = np.r_[total_t2_lab, t2_lab[i]] #(695,2)

    # 将所有的标签值的样本值叠加
    # total_pos_train = total_pos_train.astype(int)
    total_t1_lab = total_t1_lab.astype(int)
    # total_t2_lab = total_t2_lab.astype(int)

    return total_t1_lab, train1


def draw_acc_loss(train_acc_list, train_loss_list, valida_acc_list, valida_loss_list, epoch, url):
    plt.style.use("ggplot")
    # plt.rc('font', family='Times New Roman')

    plt.figure(figsize=(12, 6), dpi=500)
    accuracy = plt.subplot(121)
    plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green', label="train_acc")
    plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue', label="test_acc")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc='best')
    loss = plt.subplot(122)
    plt.plot(np.linspace(1, epoch, len(train_loss_list)), train_loss_list, color='red', label="train_loss")

    plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold', label="test_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(url + "_acc_loss.png", format='png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HSI")
    parser.add_argument('--dataset', choices=['HanChuan', 'LongKou', 'HongHu'], default='HongHu15',
                        help='dataset to use')
    parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
    parser.add_argument('--model', choices=['ViT', 'CVT', 'Cvt3D'], default='MCvT', help='model choice')
    parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='number of seed')
    parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
    parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
    parser.add_argument('--patches', type=int, default=15, help='number of patches')
    parser.add_argument('--num_classes', type=int, default=22, help='number of class')
    parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
    parser.add_argument('--epoches', type=int, default=100, help='epoch number')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    # parser.add_argument('--classes', type=int, default=22, help='category')
    args = parser.parse_args()
    # Parameter Setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("---------------------3dataset={0}---model={1}----mode={2}------------------".format(args.dataset, args.model,
                                                                                               args.mode))
    url = "/content/drive/MyDrive/HongHu/" + str(args.dataset) + "_" + str(args.model) + "_epoches_" + str(
        args.epoches) + "patches_" + str(args.patches)
    model_path = url + ".pt"
    file_name = url + ".txt"
    # load data
    # data_hsi = sio.loadmat("/home/featurize/work/Indian_pines_corrected.mat")['indian_pines_corrected']
    # gt_hsi = sio.loadmat("/home/featurize/work/Indian_pines_gt.mat")['indian_pines_gt']
    # data_hsi = pca_change(data_hsi, 100)
    # height, width, band = data_hsi.shape
    # x_train_band, y_train_band = creatCube(data_hsi, gt_hsi, windowsize=args.patches)
    # x_train_band, x_test_band, y_train_band, y_test_band = train_test_split(x_train_band, y_train_band, test_size=0.95)
    # print('x_train_band={}'.format(x_train_band.shape))
    # print('x_test_band={}'.format(x_test_band.shape))

    data_hsi = np.transpose(gdal.Open("/content/drive/MyDrive/WHU-Hi-HongHu/WHU-Hi-HongHu.tif").ReadAsArray(),
                            axes=[1, 2, 0])

    data_hsi = pca_change(data_hsi, 100)
    height, width, band = data_hsi.shape
    # 加载已经分割过的train与test
    train_meta = gdal.Open(
        "/content/drive/MyDrive/WHU-Hi-HongHu/Training samples and test samples/Train100.tif").ReadAsArray()
    test_meta = gdal.Open(
        "/content/drive/MyDrive/WHU-Hi-HongHu/Training samples and test samples/Test100.tif").ReadAsArray()
    # 在应对大数据量的样本采集时，使用下面的方法
    mirror_image = padwithzeros(data_hsi, margin=int((args.patches - 1) / 2))
    print("------------mirror_image={}------------".format(mirror_image.shape))
    # sys.exit()
    total_train, number_train = chooose_train_and_test_point(train_meta, num_classes=args.num_classes)
    print(total_train.shape)
    print("------------number_train={}------------".format(number_train))
    # sys.exit()
    # total_test, number_test = chooose_train_and_test_point(test_meta, num_classes=args.num_classes)
    total_test, number_test = divide_train_and_test_point(test_meta, num_classes=args.num_classes)
    print("------------total_test={}------------".format(total_test.shape))
    print("------------number_test={}------------".format(number_test))
    # sys.exit()
    # 此处可以将波段的组合代码（gain_neighborhood_band）删除。减少数据处理量
    x_train_band = train_and_test_data(mirror_image, data_hsi.shape[2], total_train, patch=args.patches)
    x_test_band = train_and_test_data(mirror_image, data_hsi.shape[2], total_test, patch=args.patches)
    print("x_test_band={}".format(x_test_band.shape))
    # sys.exit()
    y_train_band = train_and_test_label(number_train, num_classes=args.num_classes)
    y_test_band = train_and_test_label(number_test, num_classes=args.num_classes)

    x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor).unsqueeze(1)  # [695, 200, 7, 7]
    y_train = torch.from_numpy(y_train_band).type(torch.LongTensor)  # [695]
    Label_train = Data.TensorDataset(x_train, y_train)
    x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor).unsqueeze(1)  # [9671, 200, 7, 7]
    y_test = torch.from_numpy(y_test_band).type(torch.LongTensor)  # [9671]
    Label_test = Data.TensorDataset(x_test, y_test)

    label_train_loader = Data.DataLoader(Label_train, batch_size=32, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=32, shuffle=True)

    # img = torch.ones([1, 3, 224, 224]).to(device)
    # ----------------------------------------------------------------------
    model = CvT(image_size=args.patches, channels=band, num_classes=args.num_classes, mode=args.mode).cuda()
    # from torchsummary import summary
    #     from torchsummary import summary

    #     #输出每层网络参数信息
    #     summary(model,(1,15,15,70),batch_size=1,device="cuda")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # out = model(img)
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    tic = time.time()
    # 训练
    if args.flag_test == 'train':
        tic = time.time()
        train_acc_list = []
        valida_acc_list = []
        train_loss_list = []
        valida_loss_list = []

        for epoch in range(args.epoches):
            # scheduler.step()

            # train modeltrain
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
            scheduler.step()
            # OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
            # output_list.extend(pre_t)
            # labels_list.extend(tar_t)
            # train_acc_list.append(train_acc.cpu().numpy())
            # train_loss_list.append(train_obj.cpu().numpy())

            print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                  .format(epoch + 1, train_obj, train_acc))

            if ((epoch + 1) % args.test_freq == 0):
                model.eval()
                # print("epoch",epoch)
                with torch.no_grad():
                    test_acc2, test_obj2, tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
                    OA2, AA_mean2, Kappa2, AA2, matrix, classification = output_metric(tar_v, pre_v)
                    print("Epoch: {:03d} test_loss: {:.4f} test_acc: {:.4f} OA2: {:.4f} AA_mean2: {:.4f} Kappa2: {:.4f}"
                          .format(epoch + 1, test_obj2, test_acc2, OA2, AA_mean2, Kappa2))

                    # valida_acc_list.append(test_acc2.cpu().numpy())
                    # valida_loss_list.append(test_obj2.cpu().numpy())
        # model.eval()
        # # print("epoch",epoch)
        # with torch.no_grad():
        #     test_acc2, test_obj2, tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
        #     # OA2, AA_mean2, Kappa2, AA2, matrix, classification = output_metric(tar_v, pre_v)
        #     print("Epoch: {:03d} test_loss: {:.4f} test_acc: {:.4f}"
        #           .format(epoch + 1, test_obj2, test_acc2))
        #     valida_acc_list.append(test_acc2.cpu().numpy())
        #     valida_loss_list.append(test_obj2.cpu().numpy())
        # OA2, AA_mean2, Kappa2, AA2, matrix, classification = output_metric(tar_v, pre_v)
        # draw_acc_loss(train_acc_list, train_loss_list, valida_acc_list, valida_loss_list, args.epoches, url)
        toc = time.time()
        torch.save(model.state_dict(), model_path)
        print("Running Time: {:.2f}".format(toc - tic))
        print("**************************************************")
        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
        print(AA2)
        print()

        with open(file_name, 'w') as x_file:
            x_file.write('{:.4f} Test loss(%)'.format(test_obj2.cpu().numpy()))
            x_file.write('\n')
            x_file.write('{:.4f} Test accuracy (%)'.format(test_acc2.cpu().numpy()))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{:.4f} Kappa accuracy (%)'.format(Kappa2))
            x_file.write('\n')
            x_file.write('{:.4f} Overall accuracy (%)'.format(OA2))
            x_file.write('\n')
            x_file.write('{:.4f} Average accuracy (%)'.format(AA_mean2))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{} classification (%)'.format(classification))
            x_file.write('\n')
            x_file.write('{} matrix (%)'.format(matrix))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{:.4f} time (%)'.format(toc - tic))
        print("-------------txt完成-------------------")
        # draw_acc_loss(train_acc_list, train_loss_list, valida_acc_list, valida_loss_list, args.epoches, url)
    elif args.flag_test == 'test':

        print("Shape of out :")  # [B, num_classes]