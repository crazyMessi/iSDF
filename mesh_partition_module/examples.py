"""
Mesh Partition 使用示例
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

try:
    from mesh_partition_module import partition_mesh, visualize_partition
except ImportError:
    print("无法导入mesh_partition_module，请确保已正确安装。")
    print("可以通过在模块目录中运行'pip install -e .'来安装。")
    exit(1)

def generate_sphere_mesh(radius=1.0, num_points=1000):
    """生成一个球体网格用于测试"""
    # 使用黄金螺旋算法生成均匀分布的球面点
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(phi) * radius
    vertices = np.vstack((x, y, z)).T
    
    # 使用凸包生成三角面片
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        faces = hull.simplices
    except ImportError:
        print("警告: 未找到SciPy，使用简单面片连接。结果可能不够理想。")
        # 简单面片连接
        faces = []
        for i in range(num_points-2):
            faces.append([0, i+1, i+2])
        faces = np.array(faces)
    
    return vertices, faces

def generate_torus_mesh(R=1.0, r=0.3, n_major=50, n_minor=20):
    """生成一个环面网格用于测试"""
    # 生成参数化网格
    u = np.linspace(0, 2*np.pi, n_major)
    v = np.linspace(0, 2*np.pi, n_minor)
    U, V = np.meshgrid(u, v)
    U = U.flatten()
    V = V.flatten()
    
    # 环面参数方程
    x = (R + r * np.cos(V)) * np.cos(U)
    y = (R + r * np.cos(V)) * np.sin(U)
    z = r * np.sin(V)
    
    vertices = np.vstack((x, y, z)).T
    
    # 为了简化，我们直接使用凸包
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        faces = hull.simplices
    except ImportError:
        print("需要SciPy来生成环面网格，请使用'pip install scipy'安装。")
        return None, None
    
    return vertices, faces

def generate_cube_mesh(size=1.0):
    """生成一个立方体网格用于测试"""
    # 8个顶点坐标
    vertices = np.array([
        [-size/2, -size/2, -size/2],  # 0
        [size/2, -size/2, -size/2],   # 1
        [size/2, size/2, -size/2],    # 2
        [-size/2, size/2, -size/2],   # 3
        [-size/2, -size/2, size/2],   # 4
        [size/2, -size/2, size/2],    # 5
        [size/2, size/2, size/2],     # 6
        [-size/2, size/2, size/2]     # 7
    ])
    
    # 12个三角形面片 (每个正方形面由2个三角形组成)
    faces = np.array([
        # 底面
        [0, 1, 2], [0, 2, 3],
        # 顶面
        [4, 7, 6], [4, 6, 5],
        # 前面
        [0, 4, 5], [0, 5, 1],
        # 后面
        [3, 2, 6], [3, 6, 7],
        # 左面
        [0, 3, 7], [0, 7, 4],
        # 右面
        [1, 5, 6], [1, 6, 2]
    ])
    
    return vertices, faces

def main():
    parser = argparse.ArgumentParser(description='Mesh Partition 示例')
    parser.add_argument('--shape', type=str, default='sphere', 
                        choices=['sphere', 'torus', 'cube'],
                        help='要生成的形状类型 (默认: sphere)')
    parser.add_argument('--output', type=str, default='partition_result.png',
                        help='输出图像的路径 (默认: partition_result.png)')
    parser.add_argument('--leaf-size', type=int, default=50,
                        help='每个叶节点的目标点数，控制分区数量 (默认: 50)')
    parser.add_argument('--points', type=int, default=500,
                        help='生成的点数量 (默认: 500)')
    args = parser.parse_args()
    
    print(f"生成{args.shape}网格...")
    
    # 根据选择生成不同的形状
    if args.shape == 'sphere':
        vertices, faces = generate_sphere_mesh(radius=1.0, num_points=args.points)
    elif args.shape == 'torus':
        vertices, faces = generate_torus_mesh(R=1.0, r=0.3, n_major=args.points//20, n_minor=20)
    elif args.shape == 'cube':
        vertices, faces = generate_cube_mesh(size=1.0)
        # 增加立方体的点数
        if args.points > 8:
            # 简单地在每个面上添加随机点
            extra_points = args.points - 8
            rand_points = np.random.rand(extra_points, 3) * 2 - 1
            # 将点推到最近的面上
            for i in range(extra_points):
                # 找出最大维度并将其拉到表面
                idx = np.argmax(np.abs(rand_points[i]))
                rand_points[i, idx] = np.sign(rand_points[i, idx]) * 0.5
            
            vertices = np.vstack((vertices, rand_points))
    
    if vertices is None:
        print("生成网格失败。")
        return
    
    print(f"网格信息: {vertices.shape[0]} 个顶点, {faces.shape[0]} 个面片")
    
    # 执行网格分区
    print(f"使用叶子节点大小 {args.leaf_size} 进行网格分区...")
    regions = partition_mesh(vertices, faces, target_points_per_leaf=args.leaf_size)
    
    unique_regions = np.unique(regions)
    print(f"网格被分为 {len(unique_regions)} 个区域")
    
    # 统计每个区域的点数量
    for region in unique_regions:
        count = np.sum(regions == region)
        print(f"  区域 {region}: {count} 个点 ({count/len(regions)*100:.1f}%)")
    
    # 可视化结果
    print("生成可视化结果...")
    visualize_partition(vertices, faces, regions, output_file=args.output)

if __name__ == "__main__":
    main() 