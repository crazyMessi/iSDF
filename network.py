import torch
import torch.nn as nn
import torch.nn.functional as F
from dtype_utils import get_torch_dtype
import spconv.pytorch as spconv

class SparseVoxelEncoder(nn.Module):
    """
    将稀疏3D体素网格编码为特征向量
    使用spconv的稀疏卷积网络实现
    
    输入：NxNxN的稀疏体素网格
    输出：长度为output_dim的特征向量
    """
    def __init__(self, output_dim=1024, in_channels=1):
        super(SparseVoxelEncoder, self).__init__()
        
        # 定义稀疏卷积网络
        self.net = spconv.SparseSequential(
            # 输入层
            spconv.SparseConv3d(in_channels, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # 下采样层1
            spconv.SparseConv3d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # 下采样层2
            spconv.SparseConv3d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # 下采样层3
            spconv.SparseConv3d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # 特征提取层
            spconv.SparseConv3d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            # 全局池化
            spconv.SparseGlobalMaxPool(),
        )
        
        # 线性层，用于生成最终特征向量
        self.fc = nn.Linear(512, output_dim)
        
    def forward(self, x):
        """
        输入:
            x: spconv.SparseConvTensor - 稀疏体素网格
        输出:
            特征向量，形状为 (batch_size, output_dim)
        """
        # 应用稀疏卷积网络
        x = self.net(x)
        
        # 通过全连接层生成最终特征向量
        x = self.fc(x)  # x已经是普通Tensor，不需要.features
        
        return x
    
    @staticmethod
    def voxel_grid_to_sparse_tensor(voxel_grid, batch_size=1):
        """
        将密集的体素网格转换为稀疏张量
        
        参数:
            voxel_grid: 形状为(batch_size, N, N, N)的张量，其中非零元素表示占用的体素
            batch_size: 批次大小
            
        返回:
            spconv.SparseConvTensor - 适合输入到稀疏卷积网络的稀疏张量
        """
        device = voxel_grid.device
        spatial_shape = voxel_grid.shape[1:]
        
        indices_list = []
        features_list = []
        
        for b in range(batch_size):
            # 找出非零元素的索引
            occupied = torch.nonzero(voxel_grid[b], as_tuple=True)
            
            if len(occupied[0]) > 0:
                # 构建坐标 (batch_idx, z, y, x) - spconv使用这种顺序
                coords = torch.stack([
                    torch.full((occupied[0].shape[0],), b, device=device, dtype=torch.int32),
                    occupied[0].int(),  # z坐标
                    occupied[1].int(),  # y坐标
                    occupied[2].int()   # x坐标
                ], dim=1)
                
                # 提取特征值 (这里使用体素的值作为特征)
                feats = voxel_grid[b][occupied].unsqueeze(1)
                
                indices_list.append(coords)
                features_list.append(feats)
        
        if len(indices_list) == 0:
            # 处理空输入的情况
            indices = torch.zeros((1, 4), device=device, dtype=torch.int32)
            features = torch.zeros((1, 1), device=device)
        else:
            # 合并所有批次的坐标和特征
            indices = torch.cat(indices_list, dim=0)
            features = torch.cat(features_list, dim=0)
        
        # 创建稀疏张量
        sparse_tensor = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        
        return sparse_tensor

    @staticmethod
    def masked_voxel_grid_to_sparse_tensor(voxel_grid, mask, batch_size=1):
        """
        将带mask的体素网格转换为稀疏张量 (优化版本)
        
        参数:
            voxel_grid: 形状为(batch_size, N, N, N)的张量，包含体素值
            mask: 形状为(batch_size, N, N, N)的布尔张量，True表示有效区域
            batch_size: 批次大小
            
        返回:
            spconv.SparseConvTensor - 适合输入到稀疏卷积网络的稀疏张量
        """
        device = voxel_grid.device
        spatial_shape = voxel_grid.shape[1:]
        
        # 处理所有批次
        all_coords = []
        all_feats = []
        
        # 对每个批次并行处理
        total_valid = mask.sum().item()
        if total_valid == 0:
            # 处理空输入的情况
            indices = torch.zeros((1, 4), device=device, dtype=torch.int32)
            features = torch.zeros((1, 1), device=device)
        else:
            # 扁平化处理所有批次
            batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1, 1).expand_as(mask)
            flat_mask = mask.reshape(-1)
            flat_values = voxel_grid.reshape(-1)
            flat_batch = batch_indices.reshape(-1)
            
            # 提取有效索引
            valid_mask_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze()
            
            # 构建有效特征
            features = flat_values[valid_mask_idx].unsqueeze(1)
            
            # 构建索引矩阵
            mask_size = mask.size(1) * mask.size(2) * mask.size(3)
            z_idx = (valid_mask_idx % mask_size) // (mask.size(2) * mask.size(3))
            y_idx = (valid_mask_idx % (mask.size(2) * mask.size(3))) // mask.size(3)
            x_idx = valid_mask_idx % mask.size(3)
            b_idx = flat_batch[valid_mask_idx]
            
            indices = torch.stack([b_idx, z_idx, y_idx, x_idx], dim=1).int()
        
        # 创建稀疏张量
        sparse_tensor = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        
        return sparse_tensor

class ShapeFeatureExtractor(nn.Module):
    def __init__(self, input_channels=6, output_dim=1024):
        super(ShapeFeatureExtractor, self).__init__()
        self.layer1 = nn.Conv1d(input_channels, 64, 1)
        self.layer2 = nn.Conv1d(64, 128, 1)
        self.layer3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class SDFDecoder(nn.Module):
    def __init__(self, input_dim=1024):
        super(SDFDecoder, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
