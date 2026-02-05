"""Common activation functions and building blocks for all models"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

ACTIVATION_MAP = {
    'silu': nn.SiLU,
    'mish': nn.Mish,
    'gelu': nn.GELU,
    'leakyrelu': lambda: nn.LeakyReLU(0.1, inplace=True),
    'relu': lambda: nn.ReLU(inplace=True),
    'linear': nn.Identity
}

def get_activation_fn(name: str) -> nn.Module:
    """Returns the instantiated activation module."""
    act_fn = ACTIVATION_MAP.get(name.lower(), ACTIVATION_MAP['silu'])
    return act_fn() if callable(act_fn) else act_fn

# =============================================================================
# CORE BUILDING BLOCKS
# =============================================================================

class CoreImageBlock(nn.Module):
    """Base Conv-Norm-Act block supporting LN/BN and diverse activation"""
    def __init__(self, in_c, out_c, k_size=3, stride=1, padding=1, 
                 groups=1, use_ln=False, activation='silu', bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, k_size, stride, padding, groups=groups, bias=bias)

        if use_ln:
            self.norm = nn.LayerNorm(out_c)
        else:
            self.norm = nn.BatchNorm2d(out_c)

        self.act = get_activation_fn(activation)
        self.use_ln = use_ln

    def forward(self, x):
        x = self.conv(x)

        if self.use_ln:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.norm(x)

        return self.act(x)

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = CoreImageBlock(dim, dim, k_size=7, padding=3, 
                                     groups=dim, use_ln=True, activation='linear')
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            get_activation_fn('gelu'),
            nn.Linear(dim * 4, dim)
        )
        self.gamma = nn.Parameter(torch.ones(1) * 1e-6)

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        return (x.permute(0, 3, 1, 2) * self.gamma) + input_x

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block (MobileNetV2 style)"""
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = in_c * expand_ratio
        self.use_residual = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.append(CoreImageBlock(in_c, hidden_dim, 1, 1, 0, activation='silu'))

        layers.append(CoreImageBlock(hidden_dim, hidden_dim, 3, stride, 1, 
                                    groups=hidden_dim, activation='silu'))
        layers.append(CoreImageBlock(hidden_dim, out_c, 1, 1, 0, activation='linear'))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class GhostModule(nn.Module):
    """Ghost Module - cheap operation to reduce parameters"""
    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_channels = out_c // ratio

        self.primary_conv = CoreImageBlock(in_c, init_channels, kernel_size, 1, 
                                          kernel_size//2, activation='leakyrelu')
        self.cheap_operation = CoreImageBlock(init_channels, init_channels, dw_size, 1,
                                             dw_size//2, groups=init_channels, 
                                             activation='leakyrelu')

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class MishBottleneck(nn.Module):
    """Bottleneck with Mish activation"""
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        outplanes = planes * self.expansion

        self.conv1 = CoreImageBlock(inplanes, planes, 1, 1, 0, activation='mish')
        self.conv2 = CoreImageBlock(planes, planes, 3, stride, 1, activation='mish')
        self.conv3 = CoreImageBlock(planes, outplanes, 1, 1, 0, activation='linear')

        if stride != 1 or inplanes != outplanes:
            self.downsample = CoreImageBlock(inplanes, outplanes, 1, stride, 0, 
                                           activation='linear')
        else:
            self.downsample = nn.Identity()

        self.final_act = get_activation_fn('mish')

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += identity
        return self.final_act(out)

class CSPBlock(nn.Module):
    """CSP Block - CSPDarkNet component"""
    def __init__(self, in_c, out_c, num_blocks=3):
        super().__init__()
        mid_c = out_c // 2

        self.part1_conv = CoreImageBlock(in_c, mid_c, 1, 1, 0, activation='leakyrelu')
        self.part2_conv = CoreImageBlock(in_c, mid_c, 1, 1, 0, activation='leakyrelu')

        self.blocks = nn.Sequential(*[
            MishBottleneck(mid_c, mid_c // 4) for _ in range(num_blocks)
        ])

        self.concat_conv = CoreImageBlock(mid_c * 2, out_c, 1, 1, 0, activation='leakyrelu')

    def forward(self, x):
        part1 = self.part1_conv(x)
        part2 = self.part2_conv(x)
        part2 = self.blocks(part2)
        out = torch.cat([part1, part2], dim=1)
        return self.concat_conv(out)

class InceptionModule(nn.Module):
    """Inception Module - multi-scale feature extraction"""
    def __init__(self, in_c, out_c):
        super().__init__()
        branch_out = out_c // 4

        self.branch1 = CoreImageBlock(in_c, branch_out, 1, 1, 0, activation='silu')

        self.branch2 = nn.Sequential(
            CoreImageBlock(in_c, branch_out, 1, 1, 0, activation='silu'),
            CoreImageBlock(branch_out, branch_out, 3, 1, 1, activation='silu')
        )

        self.branch3 = nn.Sequential(
            CoreImageBlock(in_c, branch_out, 1, 1, 0, activation='silu'),
            CoreImageBlock(branch_out, branch_out, 5, 1, 2, activation='silu')
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            CoreImageBlock(in_c, branch_out, 1, 1, 0, activation='silu')
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

class DenseBlock(nn.Module):
    """Dense Block - DenseNet component"""
    def __init__(self, in_c, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in_c = in_c + i * growth_rate
            self.layers.append(nn.Sequential(
                CoreImageBlock(layer_in_c, growth_rate * 4, 1, 1, 0, activation='mish'),
                CoreImageBlock(growth_rate * 4, growth_rate, 3, 1, 1, activation='mish')
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with numerical stability improvements"""
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # CRITICAL FIX: Clip attention scores to prevent softmax overflow
        attn = torch.clamp(attn, min=-50, max=50)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            get_activation_fn('gelu'),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoderBlockWithLayerScale(nn.Module):
    """Transformer block with layer scaling for stable deep training"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.1)
        )

        # Layer scale parameters
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x

__all__ = [
    'ACTIVATION_MAP',
    'get_activation_fn',
    'CoreImageBlock',
    'ConvNeXtBlock',
    'InvertedResidualBlock',
    'GhostModule',
    'MishBottleneck',
    'CSPBlock',
    'InceptionModule',
    'DenseBlock',
    'MultiHeadSelfAttention',
    'TransformerEncoderBlock',
    'TransformerEncoderBlockWithLayerScale',
]
