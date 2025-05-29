"""
Mesh Partition Python 接口

提供更高级别的Python接口，包装原始C++模块
"""

import numpy as np

try:
    # 尝试导入编译好的C++扩展模块
    from . import process_single_mesh as _cpp_process_single_mesh
except ImportError:
    # 若导入失败，创建一个替代函数以便文档检查
    def _cpp_process_single_mesh(*args, **kwargs):
        raise ImportError("C++扩展模块未编译。请运行'pip install -e .'来编译该模块。")

def partition_mesh(vertices, faces, target_points_per_leaf=100):
    """
    将三角网格划分为区域
    
    算法步骤:
    1. 使用KD树对顶点进行均衡划分
    2. 每个叶子节点选择距离质心最近的点作为种子点
    3. 从种子点开始BFS，计算网格上的距离
    4. 将顶点分配到距离最近的区域
    
    参数:
        vertices: 形状为(N, 3)的numpy数组，表示顶点坐标
        faces: 形状为(M, 3)的numpy数组，表示面片索引
        target_points_per_leaf: 每个叶子节点的目标点数，控制分区粒度
        
    返回:
        (vertex_regions, triangle_regions)的元组，其中:
        - vertex_regions: numpy数组，长度为N，每个元素表示对应顶点所属的区域ID
        - triangle_regions: numpy数组，长度为M，每个元素表示对应三角形所属的区域ID
    """
    # 类型转换和验证
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("顶点数组必须是形状为(N, 3)的二维数组")
    
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("面片数组必须是形状为(M, 3)的二维数组")
    
    # 调用C++实现
    vertex_regions, triangle_regions = _cpp_process_single_mesh(vertices, faces, target_points_per_leaf)
    return np.array(vertex_regions), np.array(triangle_regions)

def visualize_partition(vertices, faces, regions, output_file=None, is_triangle_regions=False):
    """
    可视化网格分区结果
    
    参数:
        vertices: 形状为(N, 3)的numpy数组，表示顶点坐标
        faces: 形状为(M, 3)的numpy数组，表示面片索引
        regions: 如果is_triangle_regions=False，形状为(N,)的numpy数组，表示每个顶点的区域ID
                如果is_triangle_regions=True，形状为(M,)的numpy数组，表示每个三角形的区域ID
        output_file: 可选的输出文件路径
        is_triangle_regions: 是否visualize三角形区域，而不是顶点区域
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("可视化需要matplotlib库。请使用'pip install matplotlib'安装。")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取唯一区域ID
    unique_regions = np.unique(regions)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_regions)))
    region_colors = {region: colors[i % len(colors)] for i, region in enumerate(unique_regions)}
    
    if is_triangle_regions:
        # 如果是三角形区域，需要从三角形区域映射到点
        point_colors = np.zeros((len(vertices), 4))
        for i, face in enumerate(faces):
            region = regions[i]
            color = region_colors[region]
            # 为这个三角形的所有顶点赋予相同的颜色
            point_colors[face[0]] = color
            point_colors[face[1]] = color
            point_colors[face[2]] = color
    else:
        # 为每个顶点赋予其区域的颜色
        point_colors = np.array([region_colors[region] for region in regions])
    
    # 显示点云和区域
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=point_colors[:, :3], s=30, alpha=0.8)
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    region_type = "Triangle" if is_triangle_regions else "Vertex"
    ax.set_title(f'Mesh Partition ({len(unique_regions)} {region_type} regions)')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"可视化结果已保存至 {output_file}")
    
    plt.show()

# 暴露主要函数
__all__ = ['partition_mesh', 'visualize_partition'] 