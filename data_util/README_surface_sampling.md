# 表面采样数据加载器 (Surface Sampling DataLoader)

这个数据加载器专门用于处理三角网格数据，在网格表面均匀采样点并计算对应的法向量。它仿照了 ShapeNet 数据加载器的设计模式，并与现有的 dataloader.py 保持一致的接口风格。

## 主要功能

- **表面均匀采样**: 在三角网格表面均匀采样指定数量的点
- **法向量计算**: 自动计算采样点的表面法向量
- **网格标准化**: 可选的网格坐标标准化功能
- **数据预处理**: 支持数据预处理和缓存，提高加载效率
- **多格式支持**: 支持 .ply, .obj, .off 等常见网格文件格式
- **灵活输出**: 支持 NumPy 和 PyTorch 张量两种输出格式

## 文件结构

```
├── surface_sampling_dataloader.py  # 主要的数据加载器实现
├── example_surface_sampling.py     # 使用示例和测试代码
└── README_surface_sampling.md      # 本说明文档
```

## 安装依赖

```bash
pip install torch numpy open3d matplotlib scikit-learn tqdm
```

## 快速开始

### 基本使用

```python
from surface_sampling_dataloader import SurfaceSamplingDataset, get_surface_sampling_dataloader

# 创建数据集
dataset = SurfaceSamplingDataset(
    data_path="path/to/your/mesh/files",  # 网格文件目录
    num_samples=2048,                     # 每个网格采样的点数
    normalize=True,                       # 是否标准化网格
    save_path="processed_data/"           # 预处理数据保存路径
)

# 获取单个样本
points, normals, vertices, faces = dataset[0]
print(f"采样点形状: {points.shape}")      # [num_samples, 3]
print(f"法向量形状: {normals.shape}")     # [num_samples, 3]
print(f"原始顶点形状: {vertices.shape}")  # [num_vertices, 3]
print(f"面片形状: {faces.shape}")         # [num_faces, 3]
```

### 使用 DataLoader

```python
import torch

# 创建 PyTorch 数据加载器
dataloader = get_surface_sampling_dataloader(
    data_path="path/to/your/mesh/files",
    num_samples=2048,
    batch_size=8,
    shuffle=True,
    use_torch=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 遍历批次
for batch_idx, (points, normals, vertices, faces) in enumerate(dataloader):
    # points: [batch_size, num_samples, 3]
    # normals: [batch_size, num_samples, 3]
    # vertices: [batch_size, num_vertices, 3]
    # faces: [batch_size, num_faces, 3]
    
    print(f"批次 {batch_idx}: 采样点形状 {points.shape}")
    break
```

## 详细API说明

### SurfaceSamplingDataset

主要的数据集类，返回 NumPy 数组。

#### 参数

- `data_path` (str): 包含网格文件的目录路径
- `num_samples` (int, 默认=2048): 每个网格表面采样的点数
- `normalize` (bool, 默认=True): 是否对网格进行标准化
- `transform` (callable, 可选): 数据变换函数
- `save_path` (str, 可选): 预处理数据的保存路径

#### 返回值

`__getitem__(idx)` 返回四元组：
- `sampled_points`: 采样点坐标 [num_samples, 3]
- `sampled_normals`: 采样点法向量 [num_samples, 3]
- `vertices`: 原始网格顶点 [num_vertices, 3]
- `faces`: 三角面片索引 [num_faces, 3]

### SurfaceSamplingDatasetTorch

继承自 `SurfaceSamplingDataset`，返回 PyTorch 张量。

#### 额外参数

- `device` (str, 默认='cpu'): 张量存储设备

### get_surface_sampling_dataloader

创建数据加载器的便捷函数。

#### 参数

- `data_path` (str): 网格文件目录
- `num_samples` (int, 默认=2048): 采样点数
- `batch_size` (int, 默认=8): 批大小
- `shuffle` (bool, 默认=True): 是否打乱数据
- `num_workers` (int, 默认=4): 数据加载进程数
- `normalize` (bool, 默认=True): 是否标准化
- `save_path` (str, 可选): 预处理数据保存路径
- `use_torch` (bool, 默认=True): 是否返回PyTorch张量
- `device` (str, 默认='cpu'): 设备类型

## 高级用法

### 自定义数据变换

```python
def random_rotation_transform(points, normals, vertices, faces):
    """随机旋转变换"""
    import numpy as np
    
    # 生成随机旋转矩阵（绕Z轴）
    angle = np.random.uniform(0, 2*np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # 应用旋转到所有坐标
    points = points @ rotation_matrix.T
    normals = normals @ rotation_matrix.T
    vertices = vertices @ rotation_matrix.T
    
    return points, normals, vertices, faces

# 使用自定义变换
dataset = SurfaceSamplingDataset(
    data_path="path/to/meshes",
    num_samples=1024,
    transform=random_rotation_transform
)
```

### 数据预处理和缓存

```python
# 第一次运行时会处理所有网格并保存
dataset = SurfaceSamplingDataset(
    data_path="large_dataset/",
    num_samples=4096,
    save_path="processed_data/large_dataset/"
)

# 后续运行时会直接加载预处理的数据
dataset = SurfaceSamplingDataset(
    data_path="large_dataset/",
    num_samples=4096,
    save_path="processed_data/large_dataset/"  # 相同的保存路径
)
```

### 可视化结果

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 获取一个样本
points, normals, vertices, faces = dataset[0]

# 3D 可视化
fig = plt.figure(figsize=(12, 4))

# 原始网格顶点
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)
ax1.set_title('Original Mesh')

# 采样点
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=2)
ax2.set_title('Sampled Points')

# 采样点 + 法向量
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(points[::50, 0], points[::50, 1], points[::50, 2], s=20)
# 绘制部分法向量
for i in range(0, len(points), 50):
    start = points[i]
    end = start + normals[i] * 0.1
    ax3.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-')
ax3.set_title('Points + Normals')

plt.tight_layout()
plt.show()
```

## 运行示例

```bash
# 运行完整示例
python example_surface_sampling.py
```

示例包含：
1. 基本使用演示
2. 数据加载器批量处理
3. 自定义变换应用
4. 数据集统计信息
5. 可视化展示

## 注意事项

1. **数据路径**: 确保网格文件目录存在且包含有效的 .ply, .obj 或 .off 文件
2. **内存使用**: 大型数据集建议使用预处理和缓存功能
3. **设备选择**: 使用 PyTorch 版本时，建议根据可用硬件选择合适的设备
4. **法向量**: 如果原始网格没有法向量，会自动计算
5. **网格清理**: 会自动移除退化三角形、重复顶点等

## 与现有代码的兼容性

这个数据加载器设计时考虑了与现有 `dataloader.py` 的兼容性：

- 使用相同的 `dtype_utils` 模块来处理数据类型
- 遵循相同的预处理和缓存模式
- 支持相同的配置文件格式
- 提供类似的接口和参数名称

## 性能优化建议

1. **预处理**: 对于大型数据集，首次运行后会保存预处理结果，后续加载会更快
2. **批大小**: 根据 GPU 内存调整 `batch_size`
3. **采样数量**: 根据需要调整 `num_samples`，更多点提供更详细的表面信息但增加计算量
4. **并行加载**: 使用 `num_workers > 0` 来并行加载数据（仅限 NumPy 版本）

## 故障排除

### 常见问题

1. **"No mesh files found"**: 检查数据路径和文件格式
2. **"Empty mesh"**: 某些网格文件可能损坏或为空，会自动跳过
3. **内存不足**: 减少 `batch_size` 或 `num_samples`
4. **GPU 内存不足**: 使用 `device='cpu'` 或减少批大小

### 调试技巧

```python
# 检查数据集
dataset = SurfaceSamplingDataset("your/data/path", num_samples=100)
print(f"数据集大小: {len(dataset)}")

if len(dataset) > 0:
    points, normals, vertices, faces = dataset[0]
    print(f"采样点: {points.shape}")
    print(f"法向量: {normals.shape}")
    print(f"顶点: {vertices.shape}")
    print(f"面片: {faces.shape}")
```

## 扩展功能

可以根据需要扩展以下功能：

1. **多尺度采样**: 在同一网格上采样不同密度的点
2. **语义标签**: 如果网格有分割标签，可以添加语义信息
3. **纹理信息**: 支持纹理坐标和颜色信息
4. **曲率计算**: 添加局部曲率信息
5. **空间索引**: 添加 k-d 树等空间索引以支持邻域查询

这个数据加载器为三角网格的表面采样提供了一个灵活、高效的解决方案，可以很容易地集成到现有的深度学习项目中。 