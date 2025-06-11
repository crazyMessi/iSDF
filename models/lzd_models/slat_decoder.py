from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..trellis.modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..trellis.modules import sparse as sp
from ..trellis.models.structured_latent_vae.base import SparseTransformerBase
from ..trellis.representations import MeshExtractResult
from ..trellis.representations.mesh import SparseFeatures2Mesh
from ..trellis.models.sparse_elastic_mixin import SparseTransformerElasticMixin



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
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = super().forward(x)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        return h

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers    
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
    def convert_to_fp16(self) -> None:
        super().convert_to_fp16()
    def convert_to_fp32(self) -> None:
        super().convert_to_fp32()
    
