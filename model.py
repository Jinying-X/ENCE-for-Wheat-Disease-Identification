import os
import torch
import torch.nn as nn
from typing import Callable, Optional, List
from functools import partial
from collections import OrderedDict

# 添加Tensor导入
from torch import Tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)
        return result


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,
                 expand_c: int,
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class ImprovedCPCA(nn.Module):
    """改进的通道先验卷积注意力机制"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        # 改进的通道注意力（类似CBAM）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 多尺度空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        # 通道注意力
        ca = self.channel_att(x)
        x_ca = x * ca

        # 空间注意力（多尺度信息）
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_att(spatial_input)
        x_out = x_ca * sa

        return x_out


class ImprovedELA(nn.Module):
    """改进的高效局部注意力机制"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        reduced_channels = max(channels // reduction, 8)

        # 水平方向1D卷积
        self.conv_h = nn.Conv1d(channels, reduced_channels, 3, padding=1, bias=False)
        self.gn_h = nn.GroupNorm(1, reduced_channels)

        # 垂直方向1D卷积
        self.conv_w = nn.Conv1d(channels, reduced_channels, 3, padding=1, bias=False)
        self.gn_w = nn.GroupNorm(1, reduced_channels)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_channels * 2, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        # 水平注意力
        x_h = x.mean(dim=3)  # (b,c,h)
        x_h = self.conv_h(x_h)  # (b,c//r,h)
        x_h = self.gn_h(x_h)
        x_h = x_h.unsqueeze(-1)  # (b,c//r,h,1)

        # 垂直注意力
        x_w = x.mean(dim=2)  # (b,c,w)
        x_w = self.conv_w(x_w)  # (b,c//r,w)
        x_w = self.gn_w(x_w)
        x_w = x_w.unsqueeze(2)  # (b,c//r,1,w)

        # 融合
        attention = self.fusion(torch.cat([
            x_h.expand(-1, -1, -1, w),
            x_w.expand(-1, -1, h, -1)
        ], dim=1))

        return x * attention


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module],
                 use_cpca: bool = True,
                 use_ela: bool = True):
        super(MBConv, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.has_shortcut = (stride == 1 and input_c == out_c)
        activation_layer = nn.SiLU
        expanded_c = input_c * expand_ratio

        # 移除对expand_ratio的限制，允许expand_ratio=1
        # 当expand_ratio=1时，相当于直接进行深度卷积

        # Point-wise expansion (仅在expand_ratio != 1时使用)
        if expand_ratio != 1:
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=1,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        else:
            self.expand_conv = nn.Identity()
            expanded_c = input_c  # 当expand_ratio=1时，扩展后的通道数等于输入通道数

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # 注意力模块
        self.cpca = ImprovedCPCA(expanded_c) if use_cpca else nn.Identity()
        self.ela = ImprovedELA(expanded_c) if use_ela else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)
        self.out_channels = out_c
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        # 扩展卷积（仅在expand_ratio != 1时执行）
        if self.expand_ratio != 1:
            result = self.expand_conv(x)
        else:
            result = x  # expand_ratio=1时直接使用输入

        result = self.dwconv(result)
        result = self.se(result)

        # 应用注意力机制
        result = self.cpca(result)
        result = self.ela(result)

        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x
        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module],
                 use_ela: bool = True):
        super(FusedMBConv, self).__init__()
        assert stride in [1, 2]
        # 移除对se_ratio的限制，允许FusedMBConv也使用SE注意力
        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate
        self.has_expansion = expand_ratio != 1
        activation_layer = nn.SiLU
        expanded_c = input_c * expand_ratio

        # 扩张卷积（可选）
        if self.has_expansion:
            self.expand_conv = ConvBNAct(
                input_c,
                expanded_c,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
            # 在FusedMBConv中也支持SE注意力
            self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()
            self.ela = ImprovedELA(expanded_c) if use_ela else nn.Identity()
            self.project_conv = ConvBNAct(
                expanded_c,
                out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity
            )
        else:
            # 无扩张时直接投影
            self.project_conv = ConvBNAct(
                input_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
            self.se = nn.Identity()  # 无扩张时不使用SE
            self.ela = nn.Identity()  # 无扩张时不使用ELA

        self.out_channels = out_c
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.se(result)  # 应用SE注意力
            result = self.ela(result)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x
        return result


class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 use_svd: bool = True):
        super(EfficientNetV2, self).__init__()

        self.use_svd = use_svd

        for cnf in model_cnf:
            assert len(cnf) == 10  # 新增两个参数：use_cpca, use_ela

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []

        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            use_cpca = cnf[8] if len(cnf) > 8 else True
            use_ela = cnf[9] if len(cnf) > 9 else True

            for i in range(repeats):
                if op == MBConv:
                    layer = op(kernel_size=cnf[1],
                               input_c=cnf[4] if i == 0 else cnf[5],
                               out_c=cnf[5],
                               expand_ratio=cnf[3],
                               stride=cnf[2] if i == 0 else 1,
                               se_ratio=cnf[-1],
                               drop_rate=drop_connect_rate * block_id / total_blocks,
                               norm_layer=norm_layer,
                               use_cpca=use_cpca,
                               use_ela=use_ela)
                else:
                    layer = op(kernel_size=cnf[1],
                               input_c=cnf[4] if i == 0 else cnf[5],
                               out_c=cnf[5],
                               expand_ratio=cnf[3],
                               stride=cnf[2] if i == 0 else 1,
                               se_ratio=cnf[-1],
                               drop_rate=drop_connect_rate * block_id / total_blocks,
                               norm_layer=norm_layer,
                               use_ela=use_ela)

                if self.use_svd:
                    self._apply_svd_corrected(layer)

                blocks.append(layer)
                block_id += 1

        self.blocks = nn.Sequential(*blocks)

        # Head层
        head_input_c = model_cnf[-1][5]  # 最后一个块的输出通道
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_input_c, num_classes)
        )

    def _apply_svd_corrected(self, module):
        """修正的SVD实现"""
        for name, submodule in module.named_children():
            if isinstance(submodule, nn.Conv2d):
                weight = submodule.weight.data.clone()
                original_shape = weight.shape

                # 重塑为2D矩阵
                weight_2d = weight.view(weight.size(0), -1)

                try:
                    # SVD分解
                    U, S, V = torch.svd(weight_2d)

                    # 低秩近似
                    k = min(8, S.size(0), weight_2d.size(0), weight_2d.size(1))

                    if k > 0:
                        U_k = U[:, :k]
                        S_k = torch.diag(S[:k])
                        V_k = V[:, :k]

                        # 重建权重
                        weight_approx = U_k @ S_k @ V_k.t()
                        weight_approx = weight_approx.view(original_shape)

                        # 应用近似权重
                        submodule.weight.data = weight_approx

                except Exception as e:
                    print(f"SVD应用失败: {e}, 跳过该层")
                    continue

            elif isinstance(submodule, nn.Module) and not isinstance(submodule, (nn.BatchNorm2d, nn.GroupNorm)):
                # 递归处理子模块
                self._apply_svd_corrected(submodule)

    def apply_svd_gradually(self, current_epoch, total_epochs):
        """渐进式应用SVD"""
        if current_epoch >= total_epochs * 0.7 and self.use_svd:
            print(f"Epoch {current_epoch}: 应用SVD权重近似...")
            for module in self.blocks:
                self._apply_svd_corrected(module)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# 测试代码
if __name__ == '__main__':
    # 模型配置
    model_cnf = [
        # [repeats, kernel, stride, exp_ratio, input_c, output_c, fused, se_ratio, use_cpca, use_ela]
        [1, 3, 1, 1, 32, 16, 0, 0, True, False],
        [2, 3, 2, 4, 16, 32, 0, 0, True, False],
        [2, 5, 2, 4, 32, 48, 0, 0, False, True],
        [3, 3, 2, 4, 48, 96, 0, 0.25, True, True],
        [5, 5, 1, 6, 96, 112, 0, 0.25, True, True],
        [6, 3, 2, 6, 112, 192, 0, 0.25, True, True]
    ]

    model = EfficientNetV2(model_cnf=model_cnf, num_classes=5, use_svd=True)

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"模型输出形状: {output.shape}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters())}")
    print("模型创建成功！")