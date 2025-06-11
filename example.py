import torch
import numpy as np
from models.lzd_models.network import SparseVoxelEncoder
import time
import torch.nn as nn
import torch.nn.functional as F

class RegularVoxelEncoder(nn.Module):
    """普通3D卷积网络，用于对比"""
    def __init__(self, output_dim=1024, in_channels=1):
        super(RegularVoxelEncoder, self).__init__()
        
        self.net = nn.Sequential(
            # 输入层
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 下采样层1
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # 下采样层2
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # 下采样层3
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # 特征提取层
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        self.fc = nn.Linear(512, output_dim)
        
    def forward(self, x):
        x = self.net(x)
        x = torch.max_pool3d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_gpu_memory():
    """获取当前GPU显存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # 转换为MB
    return 0

def main():
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建一个64x64x64的体素网格
    voxel_size = 128
    batch_size = 8

    # 创建随机体素网格和mask
    voxel_grid = torch.randn((batch_size, voxel_size, voxel_size, voxel_size), device=device)
    mask = torch.zeros((batch_size, voxel_size, voxel_size, voxel_size), dtype=torch.bool, device=device)
    
    # 设置一些随机区域为有效区域
    for b in range(batch_size):
        # 随机选择一些区域设置为有效
        num_regions = np.random.randint(3, 8)
        for _ in range(num_regions):
            # 随机选择区域中心
            center = torch.randint(0, voxel_size, (3,), device=device)
            # 随机选择区域大小
            size = torch.randint(5, 15, (3,), device=device)
            # 计算区域边界
            start = torch.clamp(center - size//2, 0, voxel_size-1)
            end = torch.clamp(center + size//2, 0, voxel_size-1)
            # 设置mask
            mask[b, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = True

    print(f"体素网格形状: {voxel_grid.shape}")
    print(f"有效区域体素数量: {torch.sum(mask).item()}")

    # 初始化模型
    sparse_model = SparseVoxelEncoder(output_dim=256).to(device)
    regular_model = RegularVoxelEncoder(output_dim=256).to(device)

    # 测试稀疏卷积（使用mask）
    print("\n=== 稀疏卷积测试 (使用mask) ===")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = get_gpu_memory()
    start_time = time.time()

    # 将体素网格转换为稀疏张量
    sparse_tensor = SparseVoxelEncoder.masked_voxel_grid_to_sparse_tensor(
        voxel_grid, mask, batch_size=batch_size
    )
    print(f"稀疏张量特征数量: {sparse_tensor.features.shape}")
    print(f"稀疏张量索引数量: {sparse_tensor.indices.shape}")
    print(f"空间形状: {sparse_tensor.spatial_shape}")

    # 运行模型
    with torch.no_grad():
        sparse_features = sparse_model(sparse_tensor)

    end_time = time.time()
    end_mem = get_gpu_memory()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"稀疏卷积耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"稀疏卷积显存使用: {end_mem - start_mem:.2f}MB")
    print(f"稀疏卷积峰值显存: {peak_mem:.2f}MB")
    print(f"稀疏卷积输出形状: {sparse_features.shape}")
    print(f"稀疏卷积输出统计:")
    print(f"  均值: {sparse_features.mean().item():.4f}")
    print(f"  标准差: {sparse_features.std().item():.4f}")

    # 测试普通卷积
    print("\n=== 普通卷积测试 ===")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = get_gpu_memory()
    start_time = time.time()

    # 调整输入形状为(batch_size, channels, depth, height, width)
    regular_input = voxel_grid.unsqueeze(1)

    # 运行模型
    with torch.no_grad():
        regular_features = regular_model(regular_input)

    end_time = time.time()
    end_mem = get_gpu_memory()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"普通卷积耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"普通卷积显存使用: {end_mem - start_mem:.2f}MB")
    print(f"普通卷积峰值显存: {peak_mem:.2f}MB")
    print(f"普通卷积输出形状: {regular_features.shape}")
    print(f"普通卷积输出统计:")
    print(f"  均值: {regular_features.mean().item():.4f}")
    print(f"  标准差: {regular_features.std().item():.4f}")

    # 计算每个样本的余弦相似度
    similarities = F.cosine_similarity(sparse_features, regular_features)
    print("\n输出特征余弦相似度:")
    for i, sim in enumerate(similarities):
        print(f"  样本 {i}: {sim.item():.4f}")
    print(f"  平均相似度: {similarities.mean().item():.4f}")

if __name__ == "__main__":
    main() 