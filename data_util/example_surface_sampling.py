#!/usr/bin/env python3
"""
表面采样数据加载器使用示例

这个脚本展示了如何使用SurfaceSamplingDataset和相关的数据加载器
来处理三角网格数据，在表面均匀采样点并计算法向量。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from surface_sampling_dataloader import (
    SurfaceSamplingDataset, 
    SurfaceSamplingDatasetTorch,
    get_surface_sampling_dataloader
)


def visualize_sampled_data(points, normals, vertices, faces, title="Sampled Surface Points"):
    """
    可视化采样的点云和法向量
    """
    # 如果是PyTorch张量，转换为numpy
    if torch.is_tensor(points):
        points = points.cpu().numpy()
        normals = normals.cpu().numpy()
        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()
    
    fig = plt.figure(figsize=(15, 5))
    
    # 显示原始网格
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='lightgray', s=1, alpha=0.6)
    ax1.set_title('Original Mesh Vertices')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 显示采样点
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='red', s=2)
    ax2.set_title('Sampled Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 显示采样点和法向量
    ax3 = fig.add_subplot(133, projection='3d')
    # 选择部分点显示法向量（否则太密集）
    step = max(1, len(points) // 100)
    selected_points = points[::step]
    selected_normals = normals[::step]
    
    ax3.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], 
               c='blue', s=20)
    
    # 绘制法向量
    for i in range(len(selected_points)):
        start = selected_points[i]
        end = start + selected_normals[i] * 0.1  # 缩放法向量长度
        ax3.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                'r-', alpha=0.7, linewidth=0.5)
    
    ax3.set_title('Points with Normals')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


def save_as_point_cloud(points, normals, filename):
    """
    将采样的点和法向量保存为点云文件
    """
    if torch.is_tensor(points):
        points = points.cpu().numpy()
        normals = normals.cpu().numpy()
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # 保存点云
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


def example_basic_usage():
    """
    基本使用示例
    """
    print("=== 基本使用示例 ===")
    
    # 设置数据路径（请替换为你的实际路径）
    data_path = "data/meshes/"  # 包含.ply, .obj, .off文件的目录
    save_path = "processed_data/surface_sampling/"
    
    try:
        # 创建数据集（numpy版本）
        dataset = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=1024,  # 采样1024个点
            normalize=True,    # 标准化网格
            save_path=save_path
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 获取第一个样本
        if len(dataset) > 0:
            points, normals, vertices, faces = dataset[0]
            print(f"采样点形状: {points.shape}")
            print(f"法向量形状: {normals.shape}")
            print(f"原始顶点形状: {vertices.shape}")
            print(f"面片形状: {faces.shape}")
            
            # 可视化第一个样本
            visualize_sampled_data(points, normals, vertices, faces, 
                                 "第一个样本的表面采样结果")
            
            # 保存为点云文件
            save_as_point_cloud(points, normals, "sampled_surface_001.ply")
        
    except ValueError as e:
        print(f"错误: {e}")
        print("请确保数据路径存在且包含网格文件")


def example_dataloader_usage():
    """
    数据加载器使用示例
    """
    print("\n=== 数据加载器使用示例 ===")
    
    data_path = "data/meshes/"
    save_path = "processed_data/surface_sampling/"
    
    try:
        # 创建PyTorch数据加载器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        dataloader = get_surface_sampling_dataloader(
            data_path=data_path,
            num_samples=2048,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            normalize=True,
            save_path=save_path,
            use_torch=True,
            device=device
        )
        
        print(f"数据加载器批次数: {len(dataloader)}")
        
        # 遍历几个批次
        for batch_idx, (points, normals, vertices, faces) in enumerate(dataloader):
            print(f"\n批次 {batch_idx + 1}:")
            print(f"  采样点形状: {points.shape}")
            print(f"  法向量形状: {normals.shape}")
            print(f"  顶点形状: {vertices.shape}")
            print(f"  面片形状: {faces.shape}")
            print(f"  数据类型: {points.dtype}")
            print(f"  设备: {points.device}")
            
            # 只处理前2个批次
            if batch_idx >= 1:
                break
                
            # 可视化第一个批次的第一个样本
            if batch_idx == 0:
                visualize_sampled_data(
                    points[0], normals[0], vertices[0], faces[0],
                    f"批次 {batch_idx + 1} 第一个样本"
                )
        
    except ValueError as e:
        print(f"错误: {e}")


def example_custom_transform():
    """
    自定义变换示例
    """
    print("\n=== 自定义变换示例 ===")
    
    def random_rotation_transform(points, normals, vertices, faces):
        """
        随机旋转变换
        """
        # 生成随机旋转矩阵
        angle = np.random.uniform(0, 2*np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # 应用旋转
        points = points @ rotation_matrix.T
        normals = normals @ rotation_matrix.T
        vertices = vertices @ rotation_matrix.T
        
        return points, normals, vertices, faces
    
    data_path = "data/meshes/"
    
    try:
        # 创建带变换的数据集
        dataset = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=1024,
            normalize=True,
            transform=random_rotation_transform
        )
        
        if len(dataset) > 0:
            points, normals, vertices, faces = dataset[0]
            visualize_sampled_data(points, normals, vertices, faces,
                                 "随机旋转变换后的结果")
            
    except ValueError as e:
        print(f"错误: {e}")


def example_statistics():
    """
    数据集统计信息示例
    """
    print("\n=== 数据集统计信息 ===")
    
    data_path = "data/meshes/"
    
    try:
        dataset = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=1024,
            normalize=True
        )
        
        if len(dataset) == 0:
            print("数据集为空")
            return
            
        # 统计信息
        vertex_counts = []
        face_counts = []
        
        for i in range(min(len(dataset), 10)):  # 只处理前10个样本
            _, _, vertices, faces = dataset[i]
            vertex_counts.append(len(vertices))
            face_counts.append(len(faces))
        
        print(f"样本数量: {len(dataset)}")
        print(f"平均顶点数: {np.mean(vertex_counts):.1f}")
        print(f"平均面片数: {np.mean(face_counts):.1f}")
        print(f"顶点数范围: {min(vertex_counts)} - {max(vertex_counts)}")
        print(f"面片数范围: {min(face_counts)} - {max(face_counts)}")
        
    except ValueError as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("表面采样数据加载器示例")
    print("=" * 50)
    
    # 运行各种示例
    example_basic_usage()
    example_dataloader_usage()
    example_custom_transform()
    example_statistics()
    
    print("\n所有示例执行完成！")
    print("\n使用说明:")
    print("1. 请将网格文件放在 'data/meshes/' 目录下")
    print("2. 支持的文件格式: .ply, .obj, .off")
    print("3. 处理后的数据会保存在 'processed_data/surface_sampling/' 目录下")
    print("4. 可以通过修改参数来调整采样点数、批次大小等") 