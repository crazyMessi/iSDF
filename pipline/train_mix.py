import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lzd_models.network import ShapeFeatureExtractor, SDFDecoder
from field import SDFField
from data_util.dataloader import PointCloudDataset, QueryPointsDataset, QueryPointsDatasetMix, TriangleMeshDataset
from torch.utils.data import DataLoader
import tools
import numpy as np
import json
from data_util.dtype_utils import get_torch_dtype
import os
import open3d as o3d
from update_normal import compute_point_normals_from_mesh
import argparse
from models.lzd_models.network import SparseVoxelEncoder
import cal_wnf

# torch.autograd.set_detect_anomaly(True)

def load_config(config_dir, mode):
    """
    Load and merge configuration files.
    
    Args:
        config_dir: Directory containing configuration files
        mode: 'train' or 'test'
        
    Returns:
        Merged configuration dictionary
    """
    # Load base config
    with open(os.path.join(config_dir, 'base_config.json'), 'r') as f:
        config = json.load(f)
    
    # Load mode-specific config
    mode_config_path = os.path.join(config_dir, f'{mode}_config.json')
    if os.path.exists(mode_config_path):
        with open(mode_config_path, 'r') as f:
            mode_config = json.load(f)
            
        # Merge configurations (deep merge for nested dictionaries)
        def merge_dicts(dict1, dict2):
            """Deep merge two dictionaries"""
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        config = merge_dicts(config, mode_config)
    
    # Add mode to config
    config['mode'] = mode
    return config

def train(config):
    # 设置数据类型
    torch_dtype = get_torch_dtype(config.get("dtype", {}).get("torch", "float32"))
    
    shape_feature_dim = config["model"]["shape_encoder"]["output_dim"]
    pos_feature_dim = config["model"]["pos_encoder"]["output_dim"]
    wnf_feature_dim = config["model"]["wnf_encoder"]["output_dim"]
    shape_encoder = ShapeFeatureExtractor(input_channels=6, output_dim=shape_feature_dim)
    pos_encoder = ShapeFeatureExtractor(input_channels=3, output_dim=pos_feature_dim)
    wnf_encoder = SparseVoxelEncoder(output_dim=wnf_feature_dim)
    
    sdf_decoder = SDFDecoder(input_dim=shape_feature_dim + pos_feature_dim + wnf_feature_dim)
    
    
    shape_encoder.to(config["device"]["cuda"])
    pos_encoder.to(config["device"]["cuda"])
    sdf_decoder.to(config["device"]["cuda"])
    wnf_encoder.to(config["device"]["cuda"])
    # Initialize SDF field
    sdf_field = SDFField(resolution=config["field"]["resolution"])
    
    # Initialize optimizer
    optimizer = optim.Adam(list(shape_encoder.parameters()) + list(pos_encoder.parameters()) + list(sdf_decoder.parameters()), 
                          lr=config["training"]["learning_rate"])
    
    # Get dataset and dataloader
    triangle_mesh_dataset = TriangleMeshDataset(config["data"]["data_path"])
    
    # Create QueryPointsDataset
    sample_point_dataset = QueryPointsDataset(
        pointcloudDataset=triangle_mesh_dataset,
        resolution=config["field"]["resolution"],
        k=config["query"]["k_neighbors"],
        device=config["device"]["cuda"],
        save_path=config["data"]["save_path"]
    )
    
    # Create voxel grid
    bbox = np.array(config["field"]["bbox"])
    voxel_grid, grid_shape = tools.create_uniform_grid(resolution=config["field"]["resolution"], bbox=bbox)
    voxel_grid = torch.from_numpy(voxel_grid).to(config["device"]["cuda"]) # resolution^3 x 3
    
    mixed_sample_point_dataset = QueryPointsDatasetMix(
        query_points_dataset=sample_point_dataset,
        save_path=config["data"]["save_path"],
        device=config["device"]["cuda"]
    )
    
    sample_point_dataloader = DataLoader(
        sample_point_dataset,
        batch_size=8, # 每次处理8个点云 这个不影响结果，完全取决于内存大小
        shuffle=True,
        num_workers=4
    )
    
    query_points_dataloader = DataLoader(
        mixed_sample_point_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=config["data"]["shuffle"],
        num_workers=config["data"]["num_workers"]
    )
    
    all_points = sample_point_dataset.point_list
    gt_normals = sample_point_dataset.normal_list
    all_points = np.asarray(all_points)
    gt_normals = np.asarray(gt_normals)
    all_points = torch.from_numpy(all_points).to(config["device"]["cuda"],dtype=torch_dtype)
    gt_normals = torch.from_numpy(gt_normals).to(config["device"]["cuda"],dtype=torch_dtype)
    all_normals = torch.zeros_like(all_points)
    for i in range(all_points.shape[0]):
        tpoint = all_points[i].cpu().numpy()
        tnormal = tools.pymeshlab_normal_estimate(tpoint,10)
        tnormal = torch.from_numpy(tnormal).to(config["device"]["cuda"])
        all_normals[i] = tnormal
    
    
    WNF_calculators = []
    sp_wnf_fields = []
    for i in range(all_points.shape[0]):
        WNF_calculators.append(cal_wnf.WNF(all_points[i]))
        WNF_calculators[i].update_normal(all_normals[i])
    
    for i in range(all_points.shape[0]):
        _, _, mask, _, _ = sample_point_dataset[i]
        mask = torch.from_numpy(mask).to(config["device"]["cuda"],dtype=torch.bool)
        wnf_field = torch.zeros(grid_shape,device=config["device"]["cuda"],dtype=torch_dtype)
        wnf_field[mask.view(grid_shape)] = WNF_calculators[i].query_wn(voxel_grid[mask],batch_size=10000)
        sp_wnf_field = wnf_encoder.masked_voxel_grid_to_sparse_tensor(wnf_field.unsqueeze(0),mask.view(grid_shape).unsqueeze(0))
        sp_wnf_fields.append(sp_wnf_field)
        
    
    # SDF Training loop
    for epoch in range(config["training"]["num_epochs"]):
  
        for batch_idx, (query_points, sdistane) in enumerate(tqdm(query_points_dataloader)):
            query_points = query_points.to(config["device"]["cuda"],dtype=torch_dtype)
            sdistane = sdistane.to(config["device"]["cuda"],dtype=torch_dtype)
            # Form the point cloud - combine points and normals
            point_cloud_id = query_points[:,3] # B x 1
            point_cloud_id = point_cloud_id.long()
            point_cloud_pos = all_points[point_cloud_id] # B x N x 3
            point_cloud_normal = all_normals[point_cloud_id] # B x N x 3
            point_cloud = torch.cat([point_cloud_pos,point_cloud_normal],dim=-1) # B x N x 6
            point_cloud = point_cloud.transpose(1,2) # B x 6 x N
                        
            query_pos = query_points[:,:3] # B x 3
            query_pos = query_pos.unsqueeze(2) # B x 3 x 1
            wnf_code = []
            query_code = pos_encoder(query_pos) # B x pos_feature_dim
            
            shape_code = shape_encoder(point_cloud) # B x shape_feature_dim
            for i in range(len(point_cloud_id)):
                wnf_code.append(wnf_encoder(sp_wnf_fields[point_cloud_id[i]]))
            wnf_code = torch.cat(wnf_code,dim=0) # B x wnf_feature_dim 
            qs_feature = torch.cat([shape_code, query_code,wnf_code],dim=-1) # B x (shape_feature_dim + pos_feature_dim + wnf_feature_dim)
            
            sdf_pred = sdf_decoder(qs_feature) # B x 1
            gt_y = sdistane
            gt_y[gt_y>0] = 1
            gt_y[gt_y<0] = -1

            
            # Compute loss
            loss = nn.MSELoss()(sdf_pred.squeeze(), gt_y.squeeze())
            P = (sdf_pred>0).squeeze()
            N = (sdf_pred<0).squeeze()
            TP = (P & (gt_y>0)).sum()
            TN = (N & (gt_y<0)).sum()
            FP = (P & (gt_y<0)).sum()
            FN = (N & (gt_y>0)).sum()
            acc = (TP + TN) / (TP + TN + FP + FN)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward() # 保留计算图 因为encoder可以被更新多次
            optimizer.step()
            
            if batch_idx % config["training"]["log_interval"] == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}, Acc: {acc:.6f}')
                
        # Save checkpoint
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            checkpoint_path = f'{config["training"]["checkpoint_dir"]}/checkpoint_epoch_{epoch+1}.pt'
            os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
            torch.save({
                'shape_encoder_state_dict': shape_encoder.state_dict(),
                'pos_encoder_state_dict': pos_encoder.state_dict(),
                'sdf_decoder_state_dict': sdf_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    
            

def test(config, checkpoint_path):
    """
    Test the trained model on all point clouds in the specified directory.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to the model checkpoint
    """
    # 设置数据类型
    torch_dtype = get_torch_dtype(config.get("dtype", {}).get("torch", "float32"))
    
    # 初始化模型组件
    shape_feature_dim = config["model"]["shape_encoder"]["output_dim"]
    pos_feature_dim = config["model"]["pos_encoder"]["output_dim"]
    wnf_feature_dim = config["model"]["wnf_encoder"]["output_dim"]
    shape_encoder = ShapeFeatureExtractor(input_channels=6, output_dim=shape_feature_dim)
    pos_encoder = ShapeFeatureExtractor(input_channels=3, output_dim=pos_feature_dim)
    wnf_encoder = SparseVoxelEncoder(output_dim=wnf_feature_dim)
    sdf_decoder = SDFDecoder(input_dim=shape_feature_dim + pos_feature_dim + wnf_feature_dim)
    
    # 移动到指定设备
    device = torch.device(config["device"]["cuda"])
    shape_encoder.to(device)
    pos_encoder.to(device)
    sdf_decoder.to(device)
    wnf_encoder.to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    shape_encoder.load_state_dict(checkpoint['shape_encoder_state_dict'])
    pos_encoder.load_state_dict(checkpoint['pos_encoder_state_dict'])
    sdf_decoder.load_state_dict(checkpoint['sdf_decoder_state_dict'])
    
    # 设置模型为评估模式
    shape_encoder.eval()
    pos_encoder.eval()
    sdf_decoder.eval()
    wnf_encoder.eval()
    
    # 加载数据集
    point_cloud_dataset = PointCloudDataset(config["data"]["data_path"])
    sample_point_dataset = QueryPointsDataset(
        pointcloudDataset=point_cloud_dataset,
        resolution=config["field"]["resolution"],
        k=config["query"]["k_neighbors"],
        device=device,
        save_path=config["data"]["save_path"]
    )
    
    # 获取点云和法线
    all_points = sample_point_dataset.point_list
    all_points = np.asarray(all_points)
    all_points = torch.from_numpy(all_points).to(device, dtype=torch_dtype)
    
    # 计算法线
    all_normals = torch.zeros_like(all_points)
    for i in range(all_points.shape[0]):
        tpoint = all_points[i].cpu().numpy()
        tnormal = tools.pymeshlab_normal_estimate(tpoint, 10)
        tnormal = torch.from_numpy(tnormal).to(device)
        all_normals[i] = tnormal
    
    # 创建体素网格
    resolution = config["test"]["resolution"]
    bbox = np.array(config["field"]["bbox"])
    voxel_grid, grid_shape = tools.create_uniform_grid(resolution=resolution, bbox=bbox)
    voxel_grid = torch.from_numpy(voxel_grid).to(device)
    
    # 创建WNF计算器和计算稀疏WNF字段
    WNF_calculators = []
    sp_wnf_fields = []
    for i in range(all_points.shape[0]):
        WNF_calculators.append(cal_wnf.WNF(all_points[i]))
        WNF_calculators[i].update_normal(all_normals[i])
    
    for i in range(all_points.shape[0]):
        _, _, mask = sample_point_dataset[i]
        mask = torch.from_numpy(mask).to(device, dtype=torch.bool)
        wnf_field = torch.zeros(grid_shape, device=device, dtype=torch_dtype)
        wnf_field[mask.view(grid_shape)] = WNF_calculators[i].query_wn(voxel_grid[mask], batch_size=10000)
        sp_wnf_field = wnf_encoder.masked_voxel_grid_to_sparse_tensor(wnf_field.unsqueeze(0), mask.view(grid_shape).unsqueeze(0))
        sp_wnf_fields.append(sp_wnf_field)
    
    # 创建输出目录
    output_dir = config["test"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个点云模型
    for model_idx in range(len(all_points)):
        print(f"Processing model {model_idx+1}/{len(all_points)}")
        
        # 获取当前点云和法线
        points = all_points[model_idx]
        normals = all_normals[model_idx]
        
        # 创建用于shape_encoder的输入
        oriented_points = torch.cat([points, normals], dim=-1)
        oriented_points = oriented_points.unsqueeze(0)  # 1 x N x 6
        oriented_points = oriented_points.transpose(1, 2)  # 转置为 1 x 6 x N
        
        # 为测试生成网格点
        query_points, grid_shape = tools.create_uniform_grid(resolution=resolution, bbox=bbox)
        query_points_torch = torch.tensor(query_points, dtype=torch_dtype, device=device)
        
        # 计算每个查询点的SDF值
        sdf_values = []
        batch_size = config["test"]["batch_size"]
        num_batches = (query_points.shape[0] + batch_size - 1) // batch_size
        
        print(f"Computing SDF values for model {model_idx+1}...")
        with torch.no_grad():
            # 获取shape code
            shape_code = shape_encoder(oriented_points)  # 1 x shape_feature_dim
            
            # 获取wnf code
            wnf_code = wnf_encoder(sp_wnf_fields[model_idx])  # 1 x wnf_feature_dim
            
            for i in tqdm(range(num_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, query_points.shape[0])
                
                # 处理当前批次的查询点
                batch_query_points = query_points_torch[start_idx:end_idx].unsqueeze(2)  # B x 3 x 1
                batch_query_code = pos_encoder(batch_query_points)  # B x pos_feature_dim
                
                # 重复shape code和wnf code
                batch_shape_code = shape_code.repeat(batch_query_code.shape[0], 1)  # B x shape_feature_dim
                batch_wnf_code = wnf_code.repeat(batch_query_code.shape[0], 1)  # B x wnf_feature_dim
                
                # 连接特征
                batch_features = torch.cat([batch_shape_code, batch_query_code, batch_wnf_code], dim=1)  # B x (shape_feature_dim + pos_feature_dim + wnf_feature_dim)
                
                # 预测SDF值
                batch_sdf = sdf_decoder(batch_features).cpu().numpy()  # B x 1
                sdf_values.append(batch_sdf)
        
        # 合并所有SDF值并重塑为3D网格
        sdf_values = np.concatenate(sdf_values, axis=0).reshape(grid_shape)
        
        # 计算真实SDF值作为比较（如果需要的话）
        sdf_field = SDFField(resolution=resolution)
        gt_sdf_values = sdf_field.compute_sdf(points.cpu(), normals.cpu(), query_points)
        gt_sdf_values = gt_sdf_values.cpu().numpy()
        loss = np.linalg.norm(sdf_values.reshape(-1) - gt_sdf_values.reshape(-1))
        print(f"Loss: {loss}")
        
        # 提取等值面
        print(f"Extracting isosurface for model {model_idx+1}...")
        verts, faces = tools.extract_surface_from_scalar_field(
            scalar_field=sdf_values,
            level=0.0,
            resolution=resolution,
            bbox=bbox
        )
        
        if verts is None or faces is None:
            print(f"Failed to extract isosurface for model {model_idx+1}!")
            continue
        
        # 使用提取的网格计算点云的法线
        print(f"Computing normals from extracted mesh for model {model_idx+1}...")
        updated_normals = compute_point_normals_from_mesh(points.cpu().numpy(), verts, faces, k=10)
        
        # 创建特定于模型的输出目录
        model_output_dir = os.path.join(output_dir, f"model_{model_idx}")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 保存网格
        mesh_output_path = os.path.join(model_output_dir, "reconstructed_mesh.ply")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(mesh_output_path, mesh)
        print(f"Mesh saved to {mesh_output_path}")
        
        # 保存带有更新法线的点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(updated_normals)
        pcd_output_path = os.path.join(model_output_dir, "point_cloud_with_normals.ply")
        o3d.io.write_point_cloud(pcd_output_path, pcd)
        print(f"Point cloud with updated normals saved to {pcd_output_path}")
        
        # 可选，可视化结果
        tools.plot_partition(
            sdf_values > 0, 
            save_path=os.path.join(model_output_dir, "sdf_slice.png"),
            slice_idx=None,
            bbox=bbox,
            points=points.cpu().numpy(),
            vis=False
        )
        
        # 保存体素网格
        voxel_output_dir = config["test"]["voxel_output_dir"]
        if voxel_output_dir is not None:
            os.makedirs(voxel_output_dir, exist_ok=True)
            voxel_output_path = os.path.join(voxel_output_dir, f"voxel_grid_{model_idx}.npy")
            np.save(voxel_output_path, sdf_values)
            print(f"Voxel grid saved to {voxel_output_path}")
        
        # 可选的可视化键控
        key = False
        if key:
            import view
            view.view_data(sdf_values)
    
    print("All models processed successfully!")
    return

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or test the model')
    parser.add_argument('--config_dir', type=str, default='config', 
                        help='Directory containing configuration files')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_dir, args.mode)
    
    # Run training or testing
    if args.mode == 'train':
        train(config)
    elif args.mode == 'test':
        checkpoint_path = config["test"]["checkpoint_path"]
        test(config, checkpoint_path)