from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..trellis.modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..trellis.modules import sparse as sp
from ..trellis.models.structured_latent_vae.base import SparseTransformerBase


# 线性+layer norm+tanh lizd
class L_L_T(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(L_L_T, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)
        self.norm = nn.LayerNorm(out_features)
        self.activate = nn.Tanh()

    def forward(self, input: sp.SparseTensor) -> sp.SparseTensor:
        return input.replace(self.activate(self.norm(self.linear(input.feats))))
    
    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

# 线性+layer norm+tanh lizd
class SLatEncoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        model2latent: List[int] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        
        layers = []
        for i in range(len(model2latent)-1):
            layers.append(L_L_T(model2latent[i], model2latent[i+1]))
        layers.append(L_L_T(model2latent[-1], latent_channels * 2))
            
        
        self.model2latent_layers = nn.Sequential(*layers)
        self.out_layer = L_L_T(latent_channels * 2, latent_channels * 2) # 输出mean和logvar
        
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        # nn.init.constant_(self.out_layer.weight, 0)
        # nn.init.constant_(self.out_layer.bias, 0)
        for layer in self.model2latent_layers:
            if isinstance(layer, L_L_T):
                layer.initialize_weights()
        self.out_layer.initialize_weights()

    def forward(self, x: sp.SparseTensor, sample_posterior=True, return_raw=False):
        h = super().forward(x)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.model2latent_layers(h)
        h = self.out_layer(h)
        
        # TODO: 检查是否有用 Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z
   

'''
改编自SLAT的MeshDecoder
去掉了曲面表征和上采样的环节
'''
class SLatVoxelDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int, # qkv_dim？
        out_channels: int, # 输出的channels
        latent_channels: int, # 输入的latent channels
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        model2rep: List[int] = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        
        # self.out_layer = sp.SparseLinear(model_channels, out_channels)
        
        layers = []
        for i in range(len(model2rep)-2):
            layers.append(L_L_T(model2rep[i], model2rep[i+1]))
        self.model2rep_layers = nn.Sequential(*layers)
        assert model2rep[-1] == out_channels
        self.out_layer = nn.Linear(model2rep[-2], model2rep[-1])

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = super().forward(x)
        h = h.type(x.dtype)
        h = self.model2rep_layers(h)
        # TODO: 检查是否有用
        feats = F.normalize(h.feats)
        h = h.replace(self.out_layer(feats))
        return h

    def initialize_weights(self) -> None:
        super().initialize_weights()
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)
        for layer in self.model2rep_layers:
            if isinstance(layer, L_L_T):
                layer.initialize_weights()
    
    def convert_to_fp16(self) -> None:
        super().convert_to_fp16()
    def convert_to_fp32(self) -> None:
        super().convert_to_fp32()
    
