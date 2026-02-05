"""Factory for creating all backbone architectures"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocks import (
    ConvNeXtBlock,
    CoreImageBlock,
    CSPBlock,
    DenseBlock,
    InceptionModule,
    InvertedResidualBlock,
    MishBottleneck,
    TransformerEncoderBlockWithLayerScale,
    get_activation_fn,
)


# Handle both package imports and direct sys.path imports
try:
    from ..utils import logger
except ImportError:
    from utils import logger

# =============================================================================
# DYNAMIC CONVOLUTION
# =============================================================================

class DynamicConv(nn.Module):
    """Dynamic Convolution with per-sample kernel generation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_kernels=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_kernels = num_kernels

        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_kernels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_kernels * 2, num_kernels),
            nn.Softmax(dim=1)
        )

        self.weight_bank = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        for i in range(num_kernels):
            nn.init.kaiming_normal_(self.weight_bank[i], mode='fan_out', nonlinearity='relu')
        self.register_buffer('scale', torch.tensor((in_channels * kernel_size * kernel_size) ** -0.5))

    def forward(self, x):
        B, C_in, H, W = x.shape
        assert C_in == self.in_channels

        attention = self.controller(x)
        K = self.num_kernels
        k = self.kernel_size
        stride = self.stride
        padding = self.padding

        scale = cast(Tensor, self.scale)
        weight_bank_flat = self.weight_bank.view(K, self.out_channels, -1) * scale
        attention = attention.view(B, K, 1)
        weighted_kernels = (attention.unsqueeze(2) * weight_bank_flat.unsqueeze(0)).sum(dim=1)

        patches = F.unfold(x, kernel_size=k, padding=padding, stride=stride)
        patches = torch.clamp(patches, -10, 10)

        out_unfold = torch.bmm(weighted_kernels, patches)

        H_out = (H + 2*padding - k) // stride + 1
        W_out = (W + 2*padding - k) // stride + 1
        out = out_unfold.view(B, self.out_channels, H_out, W_out)

        out = out + self.bias.view(1, -1, 1, 1)
        return out

# =============================================================================
# SWIN TRANSFORMER COMPONENTS
# =============================================================================

def window_partition(x, window_size):
    """Partition input into non-overlapping windows"""
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_pad, W_pad = H + pad_h, W + pad_w
    else:
        H_pad, W_pad = H, W

    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    num_windows_per_image = (H // window_size) * (W // window_size)

    assert num_windows_per_image > 0
    total_windows = windows.shape[0]
    assert total_windows % num_windows_per_image == 0

    B = total_windows // num_windows_per_image

    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention with relative position bias"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Cast registered buffer to Tensor for type checking
        rel_pos_index = cast(Tensor, self.relative_position_index)
        relative_position_bias = self.relative_position_bias_table[rel_pos_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted window attention"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            get_activation_fn('gelu'),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(attn_mask == 0, 0.0)
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        mask = cast(Tensor, self.attn_mask) if self.attn_mask is not None else None
        attn_windows = self.attn(x_windows, mask=mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer - Downsamples by merging 2x2 patches"""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x

# =============================================================================
# MOBILEONE REPARAMETERIZABLE BLOCK
# =============================================================================

class MobileOneBlock(nn.Module):
    """MobileOne reparameterizable block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_conv_branches=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_conv_branches = num_conv_branches

        self.conv_kxk = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels)
            ) for _ in range(num_conv_branches)
        ])

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.has_identity = (stride == 1 and in_channels == out_channels)
        if self.has_identity:
            self.identity_bn = nn.BatchNorm2d(in_channels)

        self.activation = nn.ReLU(inplace=True)
        self.is_fused = False

        self._init_branches()

    def _init_branches(self):
        """Initialize all branches with proper scaling"""
        for branch in self.conv_kxk:
            if isinstance(branch, nn.Sequential):
                conv = cast(nn.Conv2d, branch[0])
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    conv.weight.data *= (1.0 / (self.num_conv_branches + 2))

        conv_1x1 = cast(nn.Conv2d, self.conv_1x1[0])
        nn.init.kaiming_normal_(conv_1x1.weight, mode='fan_out', nonlinearity='relu')
        with torch.no_grad():
            conv_1x1.weight.data *= (1.0 / (self.num_conv_branches + 2))

    def forward(self, x: Tensor) -> Tensor:
        if self.is_fused and hasattr(self, 'fused_conv'):
            return self.activation(self.fused_conv(x))  # type: ignore[operator]

        out = torch.zeros_like(self.conv_kxk[0](x))

        for conv in self.conv_kxk:
            out = out + conv(x)

        out = out + self.conv_1x1(x)

        if self.has_identity:
            out = out + self.identity_bn(x)

        return self.activation(out)

# =============================================================================
# BACKBONE ARCHITECTURES
# =============================================================================

class CustomConvNeXt(nn.Module):
    """Custom ConvNeXt"""
    def __init__(self, num_classes=1000):
        super().__init__()
        dims = [96, 192, 384, 768]

        self.stem = nn.Sequential(
            CoreImageBlock(3, dims[0], 4, 4, 0, activation='gelu', use_ln=True)
        )

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[i]) for _ in range([3, 3, 9, 3][i])
            ])
            self.stages.append(stage)

            if i < 3:
                self.stages.append(nn.Sequential(
                    nn.LayerNorm(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], 2, 2)
                ))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            if isinstance(stage, nn.Sequential) and isinstance(stage[0], nn.LayerNorm):
                x = x.permute(0, 2, 3, 1)
                x = stage[0](x)
                x = x.permute(0, 3, 1, 2)
                x = stage[1](x)
            else:
                x = stage(x)

        x = x.mean([-2, -1])
        x = self.norm(x.unsqueeze(1)).squeeze(1)
        return self.head(x)

class CustomEfficientNetV4(nn.Module):
    """Custom EfficientNet V4"""
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = CoreImageBlock(3, 32, 3, 2, 1, activation='silu')

        self.stages = nn.ModuleList([
            self._make_stage(32, 16, 1, 1, 1),
            self._make_stage(16, 24, 2, 2, 2),
            self._make_stage(24, 40, 2, 2, 2),
            self._make_stage(40, 80, 3, 2, 3),
            self._make_stage(80, 112, 3, 1, 3),
            self._make_stage(112, 192, 4, 2, 4),
            self._make_stage(192, 320, 1, 1, 1),
        ])

        self.head_conv = CoreImageBlock(320, 1280, 1, 1, 0, activation='silu')
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def _make_stage(self, in_c, out_c, expand_ratio, stride, num_blocks):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(InvertedResidualBlock(in_c if i == 0 else out_c,
                                               out_c, s, expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head_conv(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

class CustomSwinTransformer(nn.Module):
    """Custom Swin Transformer"""
    def __init__(self, num_classes=1000, img_size=224, patch_size=4, embed_dim=128,
                 depths=None, num_heads=None, window_size=7):
        if num_heads is None:
            num_heads = [4, 8, 16, 32]
        if depths is None:
            depths = [2, 2, 18, 2]
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_size = patch_size

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        # After stem: spatial resolution is img_size // 4
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]

        num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=0.1)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            current_resolution = (
                self.patches_resolution[0] // (2 ** i_layer),
                self.patches_resolution[1] // (2 ** i_layer)
            )

            blocks = []
            for i in range(depths[i_layer]):
                blocks.append(
                    SwinTransformerBlock(
                        dim=layer_dim,
                        input_resolution=current_resolution,
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2
                    )
                )
            self.layers.append(nn.Sequential(*blocks))

            if i_layer < self.num_layers - 1:
                self.layers.append(
                    PatchMerging(
                        input_resolution=current_resolution,
                        dim=layer_dim,
                        norm_layer=nn.LayerNorm
                    )
                )

        self.norm = nn.LayerNorm(self.num_features)

        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features * 2, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        _b, _c, _h, _w = x.shape  # Unused but documents expected shape
        x = x.flatten(2).transpose(1, 2)

        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x

class CustomDynamicConvNet(nn.Module):
    """Custom network with dynamic convolutions"""
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = CoreImageBlock(3, 64, 3, 2, 1, activation='silu')

        self.stages = nn.ModuleList([
            self._make_stage(64, 128, 3, 2),
            self._make_stage(128, 256, 4, 2),
            self._make_stage(256, 512, 6, 2),
            self._make_stage(512, 1024, 3, 2)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_c, out_c, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            if i == num_blocks - 1:
                layers.append(nn.Sequential(
                    DynamicConv(in_c if i == 0 else out_c, out_c, 3, s, 1, num_kernels=4),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                ))
            else:
                layers.append(InvertedResidualBlock(in_c if i == 0 else out_c, out_c, s, 4))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class CustomMobileOne(nn.Module):
    """Custom MobileOne"""
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.stages = nn.ModuleList([
            self._make_stage(32, 64, 2, 2),
            self._make_stage(64, 128, 3, 2),
            self._make_stage(128, 256, 4, 2),
            self._make_stage(256, 384, 2, 1)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(384, num_classes)

        self.apply(self._init_weights)

        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                for branch in m.conv_kxk:
                    if isinstance(branch, nn.Sequential):
                        conv = cast(nn.Conv2d, branch[0])
                        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                        conv.weight.data *= 0.5

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def _make_stage(self, in_c, out_c, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(MobileOneBlock(
                in_c if i == 0 else out_c,
                out_c, 3, s, 1,
                num_conv_branches=2
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def fuse_model(self) -> None:
        """Fuse reparameterization for inference"""
        for module in self.modules():
            if isinstance(module, MobileOneBlock) and hasattr(module, 'fuse_reparam'):
                module.fuse_reparam()  # type: ignore[operator]

# Simplified architectures (will be filled with actual implementations)
class GhostModuleV2Enhanced(nn.Module):
    """Enhanced Ghost Module with SE attention and residual"""
    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_size=3, use_se=True):
        super().__init__()
        init_channels = out_c // ratio

        self.primary_conv = CoreImageBlock(in_c, init_channels, kernel_size, 1,
                                          kernel_size//2, activation='leakyrelu')
        self.cheap_operation = CoreImageBlock(init_channels, init_channels, dw_size, 1,
                                             dw_size//2, groups=init_channels,
                                             activation='leakyrelu')

        # SE attention for better feature recalibration
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_c, out_c // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c // 4, out_c, 1),
                nn.Sigmoid()
            )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        if self.use_se and hasattr(self, 'se'):
            out = out * self.se(out)

        return out


class GhostBottleneckV2(nn.Module):
    """Ghost Bottleneck with residual connection and SE"""
    def __init__(self, in_c, out_c, stride=1, expand_ratio=2):
        super().__init__()
        hidden_c = in_c * expand_ratio
        self.stride = stride
        self.use_residual = stride == 1 and in_c == out_c

        # Expansion
        self.ghost1 = GhostModuleV2Enhanced(in_c, hidden_c, use_se=False)

        # Depthwise
        if stride > 1:
            self.dw = nn.Sequential(
                CoreImageBlock(hidden_c, hidden_c, 3, stride, 1,
                             groups=hidden_c, activation='leakyrelu'),
                CoreImageBlock(hidden_c, hidden_c, 3, 1, 1,
                             groups=hidden_c, activation='leakyrelu')  # Extra DW
            )
        else:
            self.dw = CoreImageBlock(hidden_c, hidden_c, 3, 1, 1,
                                    groups=hidden_c, activation='leakyrelu')

        # Projection with SE
        self.ghost2 = GhostModuleV2Enhanced(hidden_c, out_c, use_se=True)

        # Shortcut
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                CoreImageBlock(in_c, in_c, 3, stride, 1,
                             groups=in_c, activation='linear'),
                CoreImageBlock(in_c, out_c, 1, 1, 0, activation='linear')
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)
        x = self.dw(x)
        x = self.ghost2(x)

        if self.use_residual:
            return x + residual
        else:
            return x + self.shortcut(residual)


class CustomGhostNetV2(nn.Module):
    """Enhanced Custom GhostNet V2 with deeper architecture and attention"""
    def __init__(self, num_classes=1000):
        super().__init__()

        # ENHANCED: Multi-scale stem with stronger feature extraction
        self.stem = nn.Sequential(
            CoreImageBlock(3, 24, 3, 2, 1, activation='leakyrelu'),          # 224->112
            CoreImageBlock(24, 24, 3, 1, 1, groups=24, activation='leakyrelu'),  # DW
            CoreImageBlock(24, 32, 1, 1, 0, activation='leakyrelu'),         # PW expand
            CoreImageBlock(32, 32, 3, 1, 1, groups=32, activation='leakyrelu'),  # Extra DW
        )

        # ENHANCED: Better channel progression with more blocks
        self.stages = nn.ModuleList([
            # Stage 1: 112x112, 32->48
            self._make_stage(32, 48, 3, 2, expand_ratio=2),

            # Stage 2: 56x56, 48->96
            self._make_stage(48, 96, 4, 2, expand_ratio=3),

            # Stage 3: 28x28, 96->192
            self._make_stage(96, 192, 6, 2, expand_ratio=4),

            # Stage 4: 14x14, 192->384
            self._make_stage(192, 384, 5, 2, expand_ratio=4),  # Increased blocks

            # Stage 5: 7x7, 384->512 (ENHANCED - better final features)
            self._make_stage(384, 512, 4, 1, expand_ratio=3),  # Increased output channels
        ])

        # ENHANCED: Stronger head with better feature processing
        self.conv_head = nn.Sequential(
            CoreImageBlock(512, 1280, 1, 1, 0, activation='leakyrelu'),
            CoreImageBlock(1280, 1280, 3, 1, 1, groups=1280, activation='leakyrelu'),  # Extra DW
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ENHANCED: Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(1280, 640),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(640, num_classes)
        )

        # CRITICAL: Proper initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def _make_stage(self, in_c, out_c, num_blocks, stride, expand_ratio=2):
        """Create a stage with Ghost bottlenecks"""
        layers = []

        # First block handles stride
        layers.append(GhostBottleneckV2(in_c, out_c, stride, expand_ratio))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(GhostBottleneckV2(out_c, out_c, 1, expand_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.conv_head(x)
        x = self.avgpool(x).flatten(1)
        x = self.classifier(x)

        return x

class CustomResNetMish(nn.Module):
    """Custom ResNet with Mish"""
    def __init__(self, num_classes=1000, layers=None):
        if layers is None:
            layers = [3, 4, 6, 3]
        super().__init__()

        self.inplanes = 64
        self.conv1 = CoreImageBlock(3, 64, 7, 2, 3, activation='mish')
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * MishBottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(MishBottleneck(self.inplanes, planes, stride))
        self.inplanes = planes * MishBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(MishBottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class CustomCSPDarkNet(nn.Module):
    """Custom CSP DarkNet"""
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = CoreImageBlock(3, 64, 3, 1, 1, activation='leakyrelu')

        self.stages = nn.ModuleList([
            CSPBlock(64, 128, 2),
            CSPBlock(128, 256, 4),
            CSPBlock(256, 512, 8),
            CSPBlock(512, 1024, 4)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = F.max_pool2d(x, 2, 2)
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class CustomInceptionV4(nn.Module):
    """Custom Inception V4"""
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            CoreImageBlock(3, 64, 3, 2, 1, activation='silu'),
            CoreImageBlock(64, 128, 3, 1, 1, activation='silu'),
            CoreImageBlock(128, 256, 3, 2, 1, activation='silu')
        )

        self.inception_blocks = nn.Sequential(*[
            InceptionModule(256, 256) for _ in range(4)
        ])

        self.reduction = CoreImageBlock(256, 512, 3, 2, 1, activation='silu')

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_blocks(x)
        x = self.reduction(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class CustomViTHybrid(nn.Module):
    """Custom ViT Hybrid"""
    def __init__(self, num_classes=1000, img_size=224, patch_size=16):
        super().__init__()

        # DEEP CNN STEM: Progressive feature extraction (3 -> 768 channels)
        self.stem = nn.Sequential(
            # Stage 1: 224 -> 112
            CoreImageBlock(3, 64, 3, 2, 1, activation='gelu'),
            CoreImageBlock(64, 64, 3, 1, 1, activation='gelu'),

            # Stage 2: 112 -> 56
            CoreImageBlock(64, 128, 3, 2, 1, activation='gelu'),
            CoreImageBlock(128, 128, 3, 1, 1, activation='gelu'),
            CoreImageBlock(128, 128, 3, 1, 1, activation='gelu'),

            # Stage 3: 56 -> 28
            CoreImageBlock(128, 256, 3, 2, 1, activation='gelu'),
            CoreImageBlock(256, 256, 3, 1, 1, activation='gelu'),
            CoreImageBlock(256, 256, 3, 1, 1, activation='gelu'),

            # Stage 4: 28 -> 14 (deeper feature extraction)
            CoreImageBlock(256, 512, 3, 2, 1, activation='gelu'),
            CoreImageBlock(512, 512, 3, 1, 1, activation='gelu'),
            CoreImageBlock(512, 512, 3, 1, 1, activation='gelu'),

            # Final projection to ViT dimensions
            CoreImageBlock(512, 768, 1, 1, 0, activation='gelu'),
        )

        # 14x14 patches after stem
        self.num_patches = (img_size // 16) ** 2  # 196 patches
        self.embed_dim = 768  # Standard ViT-Base dimension

        # No additional projection needed - already at 768

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # DEEP TRANSFORMER: 18 blocks (ViT-Large depth)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(
                self.embed_dim,
                num_heads=12,
                mlp_ratio=4.0,
                init_values=1e-5  # Smaller init for deeper models
            )
            for _ in range(18)  # DEEP: 18 transformer blocks
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Classification head with intermediate layer
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, num_classes)
        )

        # Proper initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # Deep CNN feature extraction
        x = self.stem(x)  # (B, 768, 14, 14)

        # Convert to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Deep transformer processing
        for block in self.transformer_blocks:
            x = block(x)

        # Global pooling of CLS token
        x = self.norm(x)
        cls_output = x[:, 0]

        return self.head(cls_output)


class CustomCoAtNet(nn.Module):
    """Custom CoAtNet"""
    def __init__(self, num_classes=1000):
        super().__init__()

        # OPTIMIZED STEM: 224 -> 56
        self.stem = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Block 2: 112 -> 56
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # STAGE 1: DEEP CNN (56x56, 128 -> 256)
        stage1_blocks = []

        # Downsample: 56 -> 28
        stage1_blocks.append(InvertedResidualBlock(128, 192, 2, 6))

        # Deep feature extraction at 28x28
        stage1_blocks.extend([
            InvertedResidualBlock(192, 192, 1, 6),
            InvertedResidualBlock(192, 192, 1, 4),
            InvertedResidualBlock(192, 192, 1, 6),
            InvertedResidualBlock(192, 192, 1, 4),
            InvertedResidualBlock(192, 192, 1, 6),
        ])

        # Channel expansion
        stage1_blocks.append(InvertedResidualBlock(192, 256, 1, 6))

        self.stage1 = nn.Sequential(*stage1_blocks)

        # STAGE 2: DEEP CNN + Attention (28x28, 256 -> 512)
        stage2_cnn_blocks = []

        # Downsample: 28 -> 14
        stage2_cnn_blocks.append(InvertedResidualBlock(256, 384, 2, 6))

        # Deep CNN at 14x14 - MORE BLOCKS for better feature extraction
        stage2_cnn_blocks.extend([
            InvertedResidualBlock(384, 384, 1, 6),
            InvertedResidualBlock(384, 384, 1, 4),
            InvertedResidualBlock(384, 384, 1, 6),
            InvertedResidualBlock(384, 384, 1, 4),
            InvertedResidualBlock(384, 384, 1, 6),
            InvertedResidualBlock(384, 384, 1, 4),
        ])

        # Channel expansion to 512
        stage2_cnn_blocks.append(InvertedResidualBlock(384, 512, 1, 6))

        self.stage2_cnn = nn.Sequential(*stage2_cnn_blocks)

        # OPTIMIZED attention: 4 blocks (not 6) with better initialization
        self.stage2_attn_blocks = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(512, num_heads=16, mlp_ratio=4.0, init_values=0.1)
            for _ in range(4)  # 4 attention blocks - optimal for performance
        ])

        # STAGE 3: OPTIMIZED Transformer (14x14, 512 -> 768)
        self.stage3_proj = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1, 1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
        )

        # OPTIMAL transformer depth: 12 blocks with strong initialization
        # This balances representational power with training stability
        self.stage3_transformer = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(
                768,
                num_heads=12,
                mlp_ratio=4.0,
                init_values=0.1  # Strong initial values for stable training
            )
            for _ in range(12)  # 12 transformer blocks (was 24 - too deep!)
        ])

        self.norm = nn.LayerNorm(768)

        # Multi-layer classification head
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # Optimized stem
        x = self.stem(x)  # (B, 128, 56, 56)

        # Stage 1: Deep CNN
        x = self.stage1(x)  # (B, 256, 28, 28)

        # Stage 2: Deep CNN + Attention
        x = self.stage2_cnn(x)  # (B, 512, 14, 14)

        # Apply attention blocks with direct residuals for best gradient flow
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, 196, 512)

        for attn_block in self.stage2_attn_blocks:
            x_seq = x_seq + attn_block(x_seq)  # Direct residual connection

        x = x_seq.transpose(1, 2).reshape(b, c, h, w)

        # Stage 3: Optimized transformer
        x = self.stage3_proj(x)  # (B, 768, 14, 14)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)

        # Apply transformer blocks with direct residuals
        # Direct residuals throughout for best gradient flow
        for transformer_block in self.stage3_transformer:
            x = x + transformer_block(x)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)  # Average all token positions

        return self.head(x)


class CustomRegNet(nn.Module):
    """Custom RegNet"""
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = CoreImageBlock(3, 32, 3, 2, 1, activation='relu')

        widths = [32, 64, 160, 384]
        depths = [1, 3, 6, 6]

        self.stages = nn.ModuleList()
        in_w = 32
        for i, (w, d) in enumerate(zip(widths, depths)):
            stage = []
            for j in range(d):
                stride = 2 if j == 0 and i > 0 else 1
                stage.append(MishBottleneck(in_w if j == 0 else w * 4, w, stride))
            self.stages.append(nn.Sequential(*stage))
            in_w = w * 4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1] * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class CustomDenseNetHybrid(nn.Module):
    """Custom DenseNet Hybrid"""
    def __init__(self, num_classes=1000):
        super().__init__()
        growth_rate = 32

        self.conv1 = CoreImageBlock(3, 64, 7, 2, 3, activation='mish')
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        num_features = 64
        self.dense1 = DenseBlock(num_features, growth_rate, 6)
        num_features += growth_rate * 6
        self.trans1 = CoreImageBlock(num_features, num_features // 2, 1, 1, 0, activation='mish')
        self.pool2 = nn.AvgPool2d(2, 2)

        num_features = num_features // 2
        self.dense2 = DenseBlock(num_features, growth_rate, 12)
        num_features += growth_rate * 12
        self.trans2 = CoreImageBlock(num_features, num_features // 2, 1, 1, 0, activation='mish')
        self.pool3 = nn.AvgPool2d(2, 2)

        num_features = num_features // 2
        self.dense3 = DenseBlock(num_features, growth_rate, 8)
        num_features += growth_rate * 8

        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.pool2(x)

        x = self.dense2(x)
        x = self.trans2(x)
        x = self.pool3(x)

        x = self.dense3(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class CustomDeiTStyle(nn.Module):
    """Custom DeiT-style transformer"""
    def __init__(self, num_classes=1000, img_size=224, patch_size=16):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        embed_dim = 768  # Standard ViT-Base dimension

        # FIXED: Better patch embedding with conv stem
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, 3, 2, 1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1, bias=False),  # 56 -> 28
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 2, 1, bias=False),  # 28 -> 14
        )

        # Recalculate num_patches for conv stem
        self.num_patches = (img_size // 16) ** 2  # 14x14 = 196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))

        # FIXED: Optimal depth with layer scaling
        self.blocks = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(embed_dim, num_heads=12, mlp_ratio=4.0)
            for _ in range(12)  # Reduced from 16 to 12 for better training
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # FIXED: Separate heads with proper initialization
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        # Proper initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.trunc_normal_(self.head_dist.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.head_dist.bias)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)

        x = x + self.pos_embed

        # Apply blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Average both token predictions
        return (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2

class CustomMaxViT(nn.Module):
    """Custom MaxViT - Proper shape handling between CNN and Transformer"""
    def __init__(self, num_classes=1000):
        super().__init__()
        # ENHANCED STEM: 224 -> 56 with strong feature extraction
        self.stem = nn.Sequential(
            # 224 -> 112
            CoreImageBlock(3, 64, 3, 2, 1, activation='gelu'),
            CoreImageBlock(64, 64, 3, 1, 1, activation='gelu'),
            CoreImageBlock(64, 64, 3, 1, 1, activation='gelu'),
            # 112 -> 56
            CoreImageBlock(64, 128, 3, 2, 1, activation='gelu'),
            CoreImageBlock(128, 128, 3, 1, 1, activation='gelu'),
            CoreImageBlock(128, 128, 3, 1, 1, activation='gelu'),
        )
        # STAGE 1: DEEP CNN (56x56, 128 -> 256)
        stage1_blocks = []
        # Downsample: 56 -> 28
        stage1_blocks.append(InvertedResidualBlock(128, 192, 2, 6))
        # DEEP feature extraction at 28x28 (increased from 4 to 6)
        for _ in range(6):
            stage1_blocks.append(InvertedResidualBlock(192, 192, 1, 6))
        # Expand to 256
        stage1_blocks.append(InvertedResidualBlock(192, 256, 1, 6))
        self.stage1 = nn.Sequential(*stage1_blocks)
        # STAGE 2: DEEP CNN + Attention (28x28, 256 -> 512)
        stage2_cnn_blocks = []
        # Downsample: 28 -> 14
        stage2_cnn_blocks.append(InvertedResidualBlock(256, 384, 2, 6))
        # VERY DEEP feature extraction at 14x14 (increased from 5 to 8 blocks)
        for _ in range(8):
            stage2_cnn_blocks.append(InvertedResidualBlock(384, 384, 1, 6))
        # Expand to 512
        stage2_cnn_blocks.append(InvertedResidualBlock(384, 512, 1, 6))
        self.stage2_cnn = nn.Sequential(*stage2_cnn_blocks)
        # ENHANCED: 3 strong attention blocks (was 2)
        self.stage2_attn = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(
                512,
                num_heads=16,
                mlp_ratio=4.0,
                init_values=0.1
            )
            for _ in range(3)  # 3 attention blocks for better feature refinement
        ])
        # STAGE 3: Transformer (14x14, 512 -> 768)
        self.stage3_proj = nn.Sequential(
            CoreImageBlock(512, 768, 3, 1, 1, activation='gelu'),
            nn.Dropout2d(0.1),
        )
        # OPTIMIZED: 10 transformer blocks (was 8, but with better initialization)
        self.stage3_transformer = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(
                768,
                num_heads=12,
                mlp_ratio=4.0,
                init_values=0.1  # Strong initialization
            )
            for _ in range(10)  # 10 blocks - good balance
        ])
        # STRONG classification head
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        # Proper initialization
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x: Tensor) -> Tensor:
        # Enhanced stem
        x = self.stem(x)  # (B, 128, 56, 56)
        # Stage 1: DEEP CNN
        x = self.stage1(x)  # (B, 256, 28, 28)
        # Stage 2: VERY DEEP CNN + Attention
        x = self.stage2_cnn(x)  # (B, 512, 14, 14)
        # Apply attention with direct residuals
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, 196, 512)
        # Enhanced attention with strong residuals
        for attn_block in self.stage2_attn:
            x_seq = x_seq + attn_block(x_seq)  # Direct residual connection
        x = x_seq.transpose(1, 2).reshape(b, c, h, w)
        # Stage 3: Optimized Transformer
        x = self.stage3_proj(x)  # (B, 768, 14, 14)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
        # Apply transformer blocks with direct residuals for optimal gradient flow
        for transformer_block in self.stage3_transformer:
            x = x + transformer_block(x)  # Direct residual throughout
        x = self.norm(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.head(x)

# =============================================================================
# BACKBONE FACTORY
# =============================================================================

BACKBONE_MAP = {
    'CustomConvNeXt': CustomConvNeXt,
    'CustomEfficientNetV4': CustomEfficientNetV4,
    'CustomGhostNetV2': CustomGhostNetV2,
    'CustomResNetMish': CustomResNetMish,
    'CustomCSPDarkNet': CustomCSPDarkNet,
    'CustomInceptionV4': CustomInceptionV4,
    'CustomViTHybrid': CustomViTHybrid,
    'CustomSwinTransformer': CustomSwinTransformer,
    'CustomCoAtNet': CustomCoAtNet,
    'CustomRegNet': CustomRegNet,
    'CustomDenseNetHybrid': CustomDenseNetHybrid,
    'CustomDeiTStyle': CustomDeiTStyle,
    'CustomMaxViT': CustomMaxViT,
    'CustomMobileOne': CustomMobileOne,
    'CustomDynamicConvNet': CustomDynamicConvNet,
}

def create_custom_backbone(name: str, num_classes: int = 1000) -> nn.Module:
    """Factory function to create custom backbone"""
    if name not in BACKBONE_MAP:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONE_MAP.keys())}")

    model_class = BACKBONE_MAP[name]
    return model_class(num_classes=num_classes)

def create_custom_backbone_safe(name: str, num_classes: int = 1000) -> nn.Module | None:
    """Safe factory function with error handling.

    Returns None for invalid backbone names instead of raising an exception.
    """
    try:
        return create_custom_backbone(name, num_classes)
    except Exception as e:
        logger.error(f"Failed to create backbone {name}: {e}")
        return None

__all__ = [
    'DynamicConv',
    'WindowAttention',
    'SwinTransformerBlock',
    'PatchMerging',
    'MobileOneBlock',
    'CustomConvNeXt',
    'CustomEfficientNetV4',
    'CustomGhostNetV2',
    'CustomResNetMish',
    'CustomCSPDarkNet',
    'CustomInceptionV4',
    'CustomViTHybrid',
    'CustomSwinTransformer',
    'CustomCoAtNet',
    'CustomRegNet',
    'CustomDenseNetHybrid',
    'CustomDeiTStyle',
    'CustomMaxViT',
    'CustomMobileOne',
    'CustomDynamicConvNet',
    'BACKBONE_MAP',
    'create_custom_backbone',
    'create_custom_backbone_safe',
]
