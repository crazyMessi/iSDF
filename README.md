# 网格分区模块 (Mesh Partition)

一个基于C++的高性能网格分区库，带有Python绑定。可用于将3D三角网格划分为多个区域，适用于网格处理、形状分析和几何处理任务。

## 算法原理

该模块使用以下步骤进行网格分区：

1. 基于KD树结构对顶点进行二分划分，使每个叶子节点包含大致相同数量的顶点
2. 在每个叶子节点中，选择距离叶子节点质心最近的顶点作为种子点
3. 从所有种子点开始，使用BFS（广度优先搜索）算法计算网格上每个顶点到所有种子点的距离
4. 将每个顶点分配给距离最近的种子点所在的区域

## 安装

### 先决条件

- Python 3.6+
- C++编译器（支持C++17或C++14）
- 以下Python库：
  - numpy
  - pybind11 (>=2.5.0)
  - matplotlib (可视化时需要)

### 安装步骤

1. 克隆或下载此代码库

2. 安装依赖库：
   ```
   pip install numpy pybind11 matplotlib
   ```

3. 在项目目录中运行：
   ```
   pip install -e .
   ```

## 使用方法

### 基本用法

```python
import numpy as np
from mesh_partition import process_single_mesh

# 加载网格顶点和面片
# vertices: 形状为(N, 3)的numpy数组，表示顶点坐标
# faces: 形状为(M, 3)的numpy数组，表示面片索引

# 进行网格分区，target_points_per_leaf控制分区的粒度
regions = process_single_mesh(vertices, faces, target_points_per_leaf=50)

# regions是一个长度为N的数组，每个元素表示对应顶点的区域ID
```

### 使用测试脚本

项目附带了一个测试脚本`test_mesh_partition.py`，可以生成一个测试网格并可视化分区结果：

```
python test_mesh_partition.py
```

这将生成一个球体网格，执行分区，并显示结果。分区结果也会保存为`partition_result.png`。

### 示例输出

运行测试脚本会生成类似下面的输出：

```
生成测试网格...
网格信息: 500 个顶点, 996 个面
使用目标叶子节点大小 50 进行网格分区...
网格被分为 16 个区域
  区域 0: 45 个点 (9.0%)
  区域 1: 22 个点 (4.4%)
  ...
完成! 结果已保存到 'partition_result.png'
```

## 参数说明

`process_single_mesh`函数接受以下参数：

- `vertices`: 形状为(N, 3)的numpy数组，表示顶点坐标
- `faces`: 形状为(M, 3)的numpy数组，表示面片索引
- `target_points_per_leaf`: 控制KD树叶子节点的大小，间接控制分区数量。值越小，生成的分区越多

## 性能考虑

该模块使用C++实现核心算法，通过pybind11绑定到Python，性能高效且内存使用量低。特别适合处理大型网格。

算法的时间复杂度：
- KD树构建: O(n log n)
- BFS分区: O(n + e)，其中n是顶点数，e是边数

## 故障排除

如果在Windows上编译时出现C++17相关错误，请确保setup.py中正确设置了MSVC编译器标志：

```python
c_opts = {
    'msvc': ['/EHsc', '/std:c++17'],  # 为MSVC显式添加/std:c++17
    'unix': [],
}
``` 