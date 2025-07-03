from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..trellis.modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..trellis.modules import sparse as sp
from ..trellis.models.structured_latent_vae.base import SparseTransformerBase
from ..trellis.models.structured_latent_flow import SparseResBlock3d
from ..trellis.modules.norm import LayerNorm32



'''
相当于SparseResBlock3d，但是没有emb
'''
class SparseResConv3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def convert_to_fp16(self):
        self.apply(convert_module_to_f16)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
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
    
    def convert_to_fp16(self):
        self.apply(convert_module_to_f16)
    
    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

# 线性+layer norm+tanh lizd
class SLatEncoder(SparseTransformerBase):
    def __init__(
        self,
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
        self.out_layer = sp.SparseLinear(model_channels, latent_channels * 2)
        
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, sample_posterior=True, return_raw=False):
        h = super().forward(x)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
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
        
class VoxelGridVAE(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 num_io_res_blocks,
                 io_block_channels,
                 use_fp16,
                 model_channels,
                 use_skip_connections,
                 encoder_config,
                 decoder_config,
                 ):
        super(VoxelGridVAE, self).__init__()
        self.use_skip_connections = use_skip_connections
        self.input_dim = input_channels
        self.model_channels = model_channels
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.use_fp16 = use_fp16
        
        self.input_layer = sp.SparseLinear(input_channels,io_block_channels[0])
        self.input_blocks = nn.ModuleList([])
        for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [self.model_channels]):
            self.input_blocks.extend([
                SparseResConv3d(chs,chs) 
                for _ in range(num_io_res_blocks-1)
            ])
            self.input_blocks.append(SparseResConv3d(chs,next_chs,downsample=True))
        self.output_blocks = nn.ModuleList([])
        for chs, prev_chs in zip(reversed(io_block_channels), [self.model_channels] + list(reversed(io_block_channels))[1:]):
            self.output_blocks.append(
                SparseResConv3d(
                    prev_chs * 2 if self.use_skip_connections else prev_chs,
                    chs,
                    upsample=True
                )
            )
            self.output_blocks.extend([
                SparseResConv3d(chs*2 if self.use_skip_connections else chs,chs)
                for _ in range(num_io_res_blocks-1)
            ])
            
        self.out_layer = sp.SparseLinear(io_block_channels[0],output_channels)
                
        self.ss_encoder = SLatEncoder(**encoder_config)
        self.ss_decoder = SLatVoxelDecoder(**decoder_config)
        
        self.activation = nn.Tanh()
        
        if use_fp16:
            self.convert_to_fp16()
        
    def convert_to_fp16(self):
        for block in self.input_blocks:
            block.convert_to_fp16()
        for block in self.output_blocks:
            block.convert_to_fp16()
        self.ss_encoder.convert_to_fp16()
        self.ss_decoder.convert_to_fp16()
        self.input_layer.apply(convert_module_to_f16)
        self.out_layer.apply(convert_module_to_f16)
        
    def forward(self, x):
        h = self.input_layer(x)
        skips = []
        for block in self.input_blocks:
            h = block(h)
            skips.append(h.feats)
        h = self.ss_encoder(h)
        h = self.ss_decoder(h)
        
        for block, skip in zip(self.output_blocks, reversed(skips)):
            if self.use_skip_connections:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)))
            else:
                h = block(h)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[1:]))
        h = self.out_layer(h.type(x.dtype))
        h = self.activation(h)
        return h
        