import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mesh_partition import process_single_mesh

def generate_sphere_mesh(radius=1.0, num_points=1000):
    """生成一个简单的球体网格用于测试"""
    # 生成球面点
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    vertices = np.vstack((x, y, z)).T * radius
    
    # 使用凸包生成三角面片
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        faces = hull.simplices
    except ImportError:
        # 如果没有scipy，则生成一些简单的三角面片
        faces = []
        for i in range(num_points-2):
            faces.append([0, i+1, i+2])
        faces = np.array(faces)
    
    return vertices, faces

def visualize_partition(vertices, faces, regions):
    """可视化网格分区结果"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为每个区域生成随机颜色
    unique_regions = np.unique(regions)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_regions)))
    region_colors = {region: colors[i] for i, region in enumerate(unique_regions)}
    
    # 设置点的颜色
    point_colors = np.array([region_colors[region] for region in regions])
    
    # 显示点云和区域
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=point_colors[:, :3], s=50, alpha=0.8)
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'网格分区结果 ({len(unique_regions)} 个区域)')
    
    plt.tight_layout()
    plt.savefig('partition_result.png')
    plt.show()

def main():
    print("生成测试网格...")
    vertices, faces = generate_sphere_mesh(radius=1.0, num_points=500)
    
    print(f"网格信息: {vertices.shape[0]} 个顶点, {faces.shape[0]} 个面")
    
    # 设置叶子节点目标点数量，控制分区数量
    target_points_per_leaf = 50
    
    print(f"使用目标叶子节点大小 {target_points_per_leaf} 进行网格分区...")
    regions = process_single_mesh(vertices, faces, target_points_per_leaf)
    regions = np.array(regions)
    
    unique_regions = np.unique(regions)
    print(f"网格被分为 {len(unique_regions)} 个区域")
    
    # 统计每个区域的点数量
    for region in unique_regions:
        count = np.sum(regions == region)
        print(f"  区域 {region}: {count} 个点 ({count/len(regions)*100:.1f}%)")
    
    # 可视化结果
    print("生成可视化结果...")
    visualize_partition(vertices, faces, regions)
    print("完成! 结果已保存到 'partition_result.png'")

if __name__ == "__main__":
    main() 