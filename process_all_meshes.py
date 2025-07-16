"""
批量处理3D模型并执行网格分区

此脚本读取指定目录下的所有PLY文件，使用mesh_partition执行网格分区，
并将分区结果保存到指定输出目录。
"""

import os
import sys
import glob
import numpy as np
import argparse
from pathlib import Path
import time
import shutil

# 添加当前目录到sys.path，以确保可以导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import trimesh
    # 尝试导入独立模块
    try:
        from mesh_partition_module import process_single_mesh
    except ImportError as e:
        print(f"错误: 导入失败: {e}")
        print("请确保已安装trimesh和已编译mesh_partition_module")
        print("可以使用以下命令安装:")
        print("  pip install trimesh")
        print("  python setup_module.py build_ext --inplace")
        sys.exit(1)
                
except ImportError as e:
    print(f"错误: 导入失败: {e}")
    print("请确保已安装trimesh")
    print("可以使用以下命令安装:")
    print("  pip install trimesh")
    sys.exit(1)

def generate_random_color_map(unique_regions):
    """
    为区域生成颜色映射
    
    参数:
        unique_regions: 唯一区域ID的数组
    
    返回:
        字典，将区域ID映射到RGBA颜色值 (uint8, 范围0-255)
    """
    num_regions = len(unique_regions)
    color_map = {}
    for i, region_id in enumerate(unique_regions):
        # 生成鲜艳的随机颜色 (使用HSV空间确保颜色饱和度高)
        h = np.random.rand()  # 随机色相
        s = 0.8 + 0.2 * np.random.rand()  # 高饱和度 (0.8-1.0)
        v = 0.8 + 0.2 * np.random.rand()  # 高亮度 (0.8-1.0)
        
        # 转换HSV到RGB
        h_i = int(6 * h)
        f = 6 * h - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        # 转换到0-255范围的uint8
        color = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        color_map[region_id] = color
    return color_map


def save_segmented_mesh_optimized(mesh, regions, output_path):
    """
    以优化方式保存带有分区信息的网格，避免为每个顶点创建新的颜色数组
    
    参数:
        mesh: trimesh网格对象
        regions: 每个顶点的区域ID
        output_path: 输出文件路径
    """
    try:
        # 为每个区域生成不同的颜色
        unique_regions = np.unique(regions)
        
        # 获取颜色映射
        color_map = generate_random_color_map(unique_regions)
        
        # 创建一个新的PLY文件并直接写入
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 使用向量化操作为顶点分配颜色
        # 创建颜色数组并初始化为0
        vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
        
        # 为每个区域创建掩码，并一次性分配颜色
        for region_id, color in color_map.items():
            mask = (regions == region_id)
            vertex_colors[mask] = color
        
        # 将颜色添加到网格
        colored_mesh = mesh.copy()
        colored_mesh.visual.vertex_colors = vertex_colors
        
        # 使用PLY格式保存，因为它支持顶点颜色
        colored_mesh.export(output_path, file_type='ply')
        
        return len(unique_regions)
    except Exception as e:
        print(f"保存网格时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

def save_region_mesh(mesh, regions, region_id, output_path):
    """
    保存单个区域的网格
    
    参数:
        mesh: 原始网格
        regions: 区域ID数组
        region_id: 要保存的区域ID
        output_path: 输出文件路径
    """
    try:
        # 创建一个标记数组，只标记特定区域的顶点
        region_mask = (regions == region_id)
        
        # 如果没有顶点属于此区域，则跳过
        if not np.any(region_mask):
            return False
        
        # 提取此区域的顶点和面
        region_vertices = mesh.vertices[region_mask]
        
        # 创建一个用于重新索引的映射
        old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int32)
        old_to_new[region_mask] = np.arange(np.sum(region_mask))
        
        # 找出包含此区域顶点的面
        valid_faces = []
        for face in mesh.faces:
            # 如果面的所有顶点都在区域内，则保留该面
            if region_mask[face[0]] and region_mask[face[1]] and region_mask[face[2]]:
                # 重新索引面
                new_face = [old_to_new[face[0]], old_to_new[face[1]], old_to_new[face[2]]]
                valid_faces.append(new_face)
        
        # 如果没有有效面，则添加一些点
        if not valid_faces:
            # 创建一个点云而不是网格
            points = trimesh.points.PointCloud(vertices=region_vertices)
            points.export(output_path)
        else:
            # 创建新的网格
            region_mesh = trimesh.Trimesh(vertices=region_vertices, faces=valid_faces)
            region_mesh.export(output_path)
        
        return True
    except Exception as e:
        print(f"保存区域网格时出错: {str(e)}")
        return False

def save_triangle_region_mesh(mesh, faces, regions, region_id, output_path):
    """
    保存单个三角形区域的网格，使用向量化操作提高效率
    
    参数:
        mesh: 原始网格
        faces: 面片数组
        regions: 三角形区域ID数组
        region_id: 要保存的区域ID
        output_path: 输出文件路径
    
    返回:
        布尔值，指示是否成功保存
    """
    try:
        # 使用向量化操作查找指定区域的三角形
        region_mask = (regions == region_id)
        
        # 如果没有三角形属于此区域，则跳过
        if not np.any(region_mask):
            return False
        
        # 提取此区域的面片
        region_faces = faces[region_mask]
        
        # 找出区域中使用的所有顶点 - 使用向量化操作
        used_vertices = np.unique(region_faces.flatten())
        
        # 创建一个用于重新索引的映射 - 使用NumPy数组操作
        old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int32)
        old_to_new[used_vertices] = np.arange(len(used_vertices))
        
        # 使用向量化操作重新索引面片
        # 为了避免循环，我们将faces展平后重新索引，然后重塑回原来的形状
        new_faces = old_to_new[region_faces.flatten()].reshape(-1, 3)
        
        # 提取使用的顶点 - 向量化操作
        new_vertices = mesh.vertices[used_vertices]
        
        # 创建新的网格
        region_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        
        # 添加颜色 - 向量化操作
        color = generate_random_color_map([region_id])[region_id]
        region_mesh.visual.vertex_colors = np.tile(color, (len(new_vertices), 1))
        
        # 保存网格
        region_mesh.export(output_path)
        
        return True
    except Exception as e:
        print(f"保存三角形区域网格时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_single_ply_file(input_path, output_dir, target_points_per_leaf, 
                           save_individual_regions=True, verbose=True):
    """
    处理单个PLY文件并保存分区结果
    
    参数:
        input_path: 输入PLY文件的路径
        output_dir: 输出目录
        target_points_per_leaf: 每个叶子节点的目标点数
        save_individual_regions: 是否保存单个区域的网格
        verbose: 是否显示详细信息
    """
    start_time = time.time()
    
    # 获取模型名称
    file_name = os.path.basename(input_path)
    model_name = os.path.splitext(file_name)[0]
    
    if verbose:
        print(f"处理文件: {input_path}")
        print(f"  模型名称: {model_name}")
    
    try:
        # 加载网格
        mesh = trimesh.load(input_path)
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        if verbose:
            print(f"  顶点数量: {len(vertices)}")
            print(f"  面片数量: {len(faces)}")
        
        # 执行网格分区
        if verbose:
            print(f"  执行网格分区 (目标点数: {target_points_per_leaf})...")
        
        # 获取顶点区域和面片区域
        vertex_regions, tri_regions = process_single_mesh(vertices, faces, target_points_per_leaf)
        vertex_regions = np.array(vertex_regions)
        tri_regions = np.array(tri_regions)
        
        unique_vertex_regions = np.unique(vertex_regions)
        unique_tri_regions = np.unique(tri_regions)
        
        if verbose:
            print(f"  顶点分区数量: {len(unique_vertex_regions)}")
            print(f"  三角形分区数量: {len(unique_tri_regions)}")
        
        # 保存包含所有顶点区域的完整分区网格
        vertex_output_path = os.path.join(output_dir, f"{model_name}_vertex_regions.ply")
        if verbose:
            print(f"  保存顶点分区网格... ", end="", flush=True)
        
        start_save = time.time()
        save_segmented_mesh_optimized(mesh, vertex_regions, vertex_output_path)
        save_time = time.time() - start_save
        
        if verbose:
            print(f"完成! ({save_time:.2f}秒)")
        
        # 保存包含所有三角形区域的完整分区网格
        tri_output_path = os.path.join(output_dir, f"{model_name}_tri_regions.ply")
        if verbose:
            print(f"  保存三角形分区网格... ", end="", flush=True)
        
        # 使用向量化操作高效地基于三角形区域为顶点着色
        # 创建颜色映射
        unique_regions = np.unique(tri_regions)
        color_map = generate_random_color_map(unique_regions)
        
        # 创建颜色数组
        vertex_colors_from_tris = np.zeros((len(vertices), 4), dtype=np.uint8)
        
        # 使用累加数组统计每个顶点出现在多少个不同区域的三角形中
        vertex_region_count = np.zeros((len(vertices), len(unique_regions)), dtype=np.int32)
        
        # 使用NumPy的高级索引为每个顶点计算区域频率
        # 首先将tri_regions扩展为与面片顶点相同形状
        face_regions = np.repeat(tri_regions[:, np.newaxis], 3, axis=1)  # 形状为 [num_faces, 3]
        
        # 为所有面片顶点建立索引对应关系
        for i in range(3):
            # 为每个面片的第i个顶点更新区域计数
            face_verts = faces[:, i]  # 所有面片的第i个顶点索引
            
            # 为每个(顶点,区域)对增加计数
            for r_idx, r_id in enumerate(unique_regions):
                mask = (tri_regions == r_id)
                verts_to_update = face_verts[mask]
                for v in verts_to_update:
                    vertex_region_count[v, r_idx] += 1
        
        # 为每个顶点找出最常见的区域
        most_common_region_idx = np.argmax(vertex_region_count, axis=1)
        most_common_region_id = unique_regions[most_common_region_idx]
        
        # 一次性为所有顶点分配颜色
        for i, region_id in enumerate(most_common_region_id):
            vertex_colors_from_tris[i] = color_map[region_id]
            
        # 将颜色添加到网格
        tri_colored_mesh = mesh.copy()
        tri_colored_mesh.visual.vertex_colors = vertex_colors_from_tris
        tri_colored_mesh.export(tri_output_path, file_type='ply')
        
        save_time = time.time() - start_save
        if verbose:
            print(f"完成! ({save_time:.2f}秒)")
        
        # 如果需要，为每个区域创建单独的网格文件
        if save_individual_regions:
            if verbose:
                print(f"  保存单个区域网格...")
            
            # 顶点区域 - 使用向量化操作
            vertex_region_stats = []
            for i, region_id in enumerate(unique_vertex_regions):
                if verbose and i % 50 == 0:
                    print(f"    处理顶点区域 {i+1}/{len(unique_vertex_regions)}...", flush=True)
                
                # 创建区域特定的输出路径
                output_path = os.path.join(output_dir, f"{model_name}_vertex_{region_id}.ply")
                
                # 找出此区域的所有顶点
                region_mask = vertex_regions == region_id
                vertex_count = np.sum(region_mask)
                percentage = (vertex_count / len(vertices)) * 100
                
                # 只保存具有足够顶点的区域
                if vertex_count > 0 and save_region_mesh(mesh, vertex_regions, region_id, output_path):
                    vertex_region_stats.append((region_id, vertex_count, percentage))
            
            # 三角形区域 - 使用向量化操作
            tri_region_stats = []
            for i, region_id in enumerate(unique_tri_regions):
                if verbose and i % 50 == 0:
                    print(f"    处理三角形区域 {i+1}/{len(unique_tri_regions)}...", flush=True)
                
                # 创建区域特定的输出路径
                output_path = os.path.join(output_dir, f"{model_name}_tri_{region_id}.ply")
                
                # 计算此区域的三角形数量
                tri_mask = tri_regions == region_id
                tri_count = np.sum(tri_mask)
                percentage = (tri_count / len(faces)) * 100
                
                # 保存包含这个区域三角形的网格
                if save_triangle_region_mesh(mesh, faces, tri_regions, region_id, output_path):
                    tri_region_stats.append((region_id, tri_count, percentage))
            
            if verbose:
                print(f"  顶点区域统计:")
                for region_id, vertex_count, percentage in vertex_region_stats:
                    print(f"    区域 {region_id}: {vertex_count} 顶点 ({percentage:.1f}%)")
                
                print(f"  三角形区域统计:")
                for region_id, tri_count, percentage in tri_region_stats:
                    print(f"    区域 {region_id}: {tri_count} 三角形 ({percentage:.1f}%)")
        
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"  完成! 总耗时: {elapsed_time:.2f} 秒")
        
        return True
    
    except Exception as e:
        print(f"处理文件 {input_path} 时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_dir, target_points_per_leaf, 
                     save_individual_regions=True, pattern="*.ply", verbose=True):
    """
    处理目录中的所有匹配文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        target_points_per_leaf: 每个叶子节点的目标点数
        save_individual_regions: 是否保存单个区域的网格
        pattern: 文件匹配模式
        verbose: 是否显示详细信息
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有匹配的文件
    file_pattern = os.path.join(input_dir, pattern)
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"在 {input_dir} 中没有找到与 {pattern} 匹配的文件")
        return
    
    print(f"找到 {len(files)} 个文件待处理...")
    
    # 处理每个文件
    success_count = 0
    for i, file_path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] ", end="")
        if process_single_ply_file(
            file_path, 
            output_dir, 
            target_points_per_leaf, 
            save_individual_regions, 
            verbose
        ):
            success_count += 1
    
    print(f"\n处理完成! 成功: {success_count}/{len(files)}")

import json
def main():
    config = json.load(open("config/local_config.json"))
    dafault_input_dir = config.get("origin_mesh_path")
    dataset_name = os.path.basename(dafault_input_dir)
    dafault_output_dir = config.get("database_path") + dataset_name + "_mesh_segment_" + str(config.get("leaf_size"))
    
    
    parser = argparse.ArgumentParser(description='批量处理PLY文件并执行网格分区')
    parser.add_argument('--input-dir', type=str, default=dafault_input_dir,
                        help='输入PLY文件目录 (默认: ' + dafault_input_dir + ')')
    parser.add_argument('--output-dir', type=str, default=dafault_output_dir,
                        help='输出分区网格目录 (默认: ' + dafault_output_dir + ')')
    parser.add_argument('--leaf-size', type=int, default=config.get("leaf_size"),
                        help='每个叶子节点的目标点数 (默认: 10000)。当一块曲面内的点数小于此值时，将不再进行分区')
    parser.add_argument('--pattern', type=str, default="*.ply",
                        help='文件匹配模式 (默认: *.ply)')
    parser.add_argument('--no-individual-regions', action='store_true',
                        help='不保存单个区域的网格文件，只保存包含所有区域的网格')
    parser.add_argument('--quiet', action='store_true',
                        help='减少输出信息')
    
    args = parser.parse_args()
    
    print("网格分区批处理")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"文件模式: {args.pattern}")
    print(f"叶子节点大小: {args.leaf_size}")
    print(f"保存单个区域: {'否' if args.no_individual_regions else '是'}")
    
    process_directory(
        args.input_dir,
        args.output_dir,
        args.leaf_size,
        not args.no_individual_regions,
        args.pattern,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main() 