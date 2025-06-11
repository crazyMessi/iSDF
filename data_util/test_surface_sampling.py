#!/usr/bin/env python3
"""
表面采样数据加载器测试脚本

用于快速测试 SurfaceSamplingDataset 的基本功能
"""

import os
import numpy as np
import open3d as o3d
from surface_sampling_dataloader import SurfaceSamplingDataset, get_surface_sampling_dataloader


def create_test_mesh(save_path="test_data/"):
    """
    创建一个简单的测试网格（球体）用于测试
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 创建球体网格
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()
    
    # 保存为 PLY 文件
    mesh_file = os.path.join(save_path, "test_sphere.ply")
    o3d.io.write_triangle_mesh(mesh_file, mesh)
    
    # 创建立方体网格
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube.compute_vertex_normals()
    cube_file = os.path.join(save_path, "test_cube.ply")
    o3d.io.write_triangle_mesh(cube_file, cube)
    
    print(f"测试网格已创建在: {save_path}")
    return save_path


def test_basic_dataset():
    """
    测试基本数据集功能
    """
    print("=== 测试基本数据集功能 ===")
    
    # 创建测试数据
    data_path = create_test_mesh()
    
    try:
        # 创建数据集
        dataset = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=512,
            normalize=True
        )
        
        print(f"✓ 数据集创建成功，包含 {len(dataset)} 个网格")
        
        if len(dataset) > 0:
            # 测试第一个样本
            points, normals, vertices, faces = dataset[0]
            
            print(f"✓ 数据形状检查:")
            print(f"  - 采样点: {points.shape}")
            print(f"  - 法向量: {normals.shape}")
            print(f"  - 原始顶点: {vertices.shape}")
            print(f"  - 面片: {faces.shape}")
            
            # 验证数据类型
            assert points.dtype == np.float32, f"采样点数据类型错误: {points.dtype}"
            assert normals.dtype == np.float32, f"法向量数据类型错误: {normals.dtype}"
            assert faces.dtype == np.int32, f"面片数据类型错误: {faces.dtype}"
            print("✓ 数据类型检查通过")
            
            # 验证形状
            assert points.shape == (512, 3), f"采样点形状错误: {points.shape}"
            assert normals.shape == (512, 3), f"法向量形状错误: {normals.shape}"
            assert vertices.shape[1] == 3, f"顶点形状错误: {vertices.shape}"
            assert faces.shape[1] == 3, f"面片形状错误: {faces.shape}"
            print("✓ 数据形状检查通过")
            
            # 验证法向量归一化
            normal_lengths = np.linalg.norm(normals, axis=1)
            assert np.allclose(normal_lengths, 1.0, atol=1e-5), "法向量未正确归一化"
            print("✓ 法向量归一化检查通过")
            
            # 验证数据范围（标准化后应该在 [-1, 1] 范围内）
            assert np.all(points >= -1.5) and np.all(points <= 1.5), "标准化后的点超出合理范围"
            print("✓ 数据范围检查通过")
            
        return True
        
    except Exception as e:
        print(f"✗ 基本数据集测试失败: {str(e)}")
        return False


def test_dataloader():
    """
    测试数据加载器功能
    """
    print("\n=== 测试数据加载器功能 ===")
    
    data_path = "test_data/"
    
    try:
        # 测试 NumPy 版本
        dataloader_np = get_surface_sampling_dataloader(
            data_path=data_path,
            num_samples=256,
            batch_size=2,
            shuffle=False,
            use_torch=False
        )
        
        print(f"✓ NumPy 数据加载器创建成功，批次数: {len(dataloader_np)}")
        
        # 测试一个批次
        for batch_idx, (points, normals, vertices, faces) in enumerate(dataloader_np):
            print(f"✓ NumPy 批次 {batch_idx} 数据形状:")
            print(f"  - 采样点: {points.shape}")
            print(f"  - 法向量: {normals.shape}")
            assert len(points.shape) == 3, "批次数据应该是3维"
            assert points.shape[1] == 256, f"采样点数错误: {points.shape[1]}"
            break
        
        # 测试 PyTorch 版本
        try:
            import torch
            dataloader_torch = get_surface_sampling_dataloader(
                data_path=data_path,
                num_samples=256,
                batch_size=2,
                shuffle=False,
                use_torch=True,
                device='cpu'
            )
            
            print(f"✓ PyTorch 数据加载器创建成功")
            
            # 测试一个批次
            for batch_idx, (points, normals, vertices, faces) in enumerate(dataloader_torch):
                print(f"✓ PyTorch 批次 {batch_idx} 数据:")
                print(f"  - 采样点: {points.shape}, 类型: {type(points)}")
                print(f"  - 设备: {points.device}")
                assert torch.is_tensor(points), "应该返回 PyTorch 张量"
                break
                
        except ImportError:
            print("⚠ PyTorch 未安装，跳过 PyTorch 测试")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {str(e)}")
        return False


def test_preprocessing_cache():
    """
    测试预处理和缓存功能
    """
    print("\n=== 测试预处理和缓存功能 ===")
    
    data_path = "test_data/"
    save_path = "test_processed/"
    
    try:
        # 第一次创建（会进行预处理）
        print("第一次创建数据集（预处理）...")
        dataset1 = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=128,
            save_path=save_path
        )
        
        # 检查预处理文件是否创建
        assert os.path.exists(save_path + "sampled_points.npy"), "预处理文件未创建"
        print("✓ 预处理文件创建成功")
        
        # 第二次创建（应该加载缓存）
        print("第二次创建数据集（加载缓存）...")
        dataset2 = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=128,
            save_path=save_path
        )
        
        # 验证两次创建的结果一致
        if len(dataset1) > 0 and len(dataset2) > 0:
            points1, _, _, _ = dataset1[0]
            points2, _, _, _ = dataset2[0]
            assert np.allclose(points1, points2), "缓存加载的数据不一致"
            print("✓ 缓存数据一致性检查通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 预处理和缓存测试失败: {str(e)}")
        return False


def test_mesh_normalization():
    """
    测试网格标准化功能
    """
    print("\n=== 测试网格标准化功能 ===")
    
    data_path = "test_data/"
    
    try:
        # 测试标准化版本
        dataset_norm = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=256,
            normalize=True
        )
        
        # 测试非标准化版本
        dataset_no_norm = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=256,
            normalize=False
        )
        
        if len(dataset_norm) > 0:
            points_norm, _, vertices_norm, _ = dataset_norm[0]
            points_no_norm, _, vertices_no_norm, _ = dataset_no_norm[0]
            
            # 标准化版本的数据应该在更小的范围内
            norm_range = np.ptp(points_norm)  # 极差
            no_norm_range = np.ptp(points_no_norm)
            
            print(f"标准化版本数据范围: {norm_range:.3f}")
            print(f"非标准化版本数据范围: {no_norm_range:.3f}")
            
            # 标准化版本应该接近 [-1, 1] 的范围
            assert norm_range < 3.0, "标准化后的数据范围过大"
            print("✓ 网格标准化功能正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 网格标准化测试失败: {str(e)}")
        return False


def cleanup_test_files():
    """
    清理测试文件
    """
    import shutil
    
    test_dirs = ["test_data/", "test_processed/"]
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    test_files = ["sampled_surface_001.ply"]
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print("✓ 测试文件清理完成")


def main():
    """
    运行所有测试
    """
    print("开始运行表面采样数据加载器测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("基本数据集功能", test_basic_dataset()))
    test_results.append(("数据加载器功能", test_dataloader()))
    test_results.append(("预处理和缓存", test_preprocessing_cache()))
    test_results.append(("网格标准化", test_mesh_normalization()))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    # 清理测试文件
    cleanup_test_files()
    
    if passed == total:
        print("🎉 所有测试通过！数据加载器工作正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 