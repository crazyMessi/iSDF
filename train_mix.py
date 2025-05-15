import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from network import ShapeFeatureExtractor, SDFDecoder
from field import SDFField
from dataloader import PointCloudDataset, QueryPointsDataset, QueryPointsDatasetMix
from torch.utils.data import DataLoader
import tools
import numpy as np
import json
from dtype_utils import get_torch_dtype
import os
import open3d as o3d
from update_normal import compute_point_normals_from_mesh
import argparse

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
    shape_encoder = ShapeFeatureExtractor(input_channels=6, output_dim=shape_feature_dim)
    pos_encoder = ShapeFeatureExtractor(input_channels=3, output_dim=pos_feature_dim)
    sdf_decoder = SDFDecoder(input_dim=shape_feature_dim + pos_feature_dim)
    
    shape_encoder.to(config["device"]["cuda"])
    pos_encoder.to(config["device"]["cuda"])
    sdf_decoder.to(config["device"]["cuda"])
    
    # Initialize SDF field
    sdf_field = SDFField(resolution=config["field"]["resolution"])
    
    # Initialize optimizer
    optimizer = optim.Adam(list(shape_encoder.parameters()) + list(pos_encoder.parameters()) + list(sdf_decoder.parameters()), 
                          lr=config["training"]["learning_rate"])
    
    # Get dataset and dataloader
    point_cloud_dataset = PointCloudDataset(config["data"]["data_path"])
    
    # Create QueryPointsDataset
    query_dataset = QueryPointsDataset(
        pointcloudDataset=point_cloud_dataset,
        resolution=config["field"]["resolution"],
        k=config["query"]["k_neighbors"],
        device=config["device"]["cuda"],
        save_path=config["data"]["save_path"]
    )
    
    # Create voxel grid
    bbox = np.array(config["field"]["bbox"])
    voxel_grid, shape = tools.create_uniform_grid(resolution=config["field"]["resolution"], bbox=bbox)
    voxel_grid = torch.from_numpy(voxel_grid).to(config["device"]["cuda"]) # resolution^3 x 3
    
    query_points_dataset = QueryPointsDatasetMix(
        query_points_dataset=query_dataset,
        save_path=config["data"]["save_path"],
        device=config["device"]["cuda"]
    )
    
    dataloader = DataLoader(
        query_points_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=config["data"]["shuffle"],
        num_workers=config["data"]["num_workers"]
    )
    
    all_points = query_dataset.point_list
    gt_normals = query_dataset.normal_list
    all_points = np.asarray(all_points)
    gt_normals = np.asarray(gt_normals)
    all_points = torch.from_numpy(all_points).to(config["device"]["cuda"])
    gt_normals = torch.from_numpy(gt_normals).to(config["device"]["cuda"])
    
    all_normals = torch.zeros_like(all_points)
    for i in range(all_points.shape[0]):
        tpoint = all_points[i].cpu().numpy()
        tnormal = tools.pymeshlab_normal_estimate(tpoint,10)
        tnormal = torch.from_numpy(tnormal).to(config["device"]["cuda"])
        all_normals[i] = tnormal
    
    
    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        for batch_idx, (query_points, sdistane) in enumerate(tqdm(dataloader)):
            # Form the point cloud - combine points and normals
            point_cloud_id = query_points[:,3] # B x 1
            point_cloud_id = point_cloud_id.long()
            point_cloud_pos = all_points[point_cloud_id] # B x N x 3
            point_cloud_normal = all_normals[point_cloud_id] # B x N x 3
            
            
            point_cloud = torch.cat([point_cloud_pos, point_cloud_normal],dim=-1) # B x N x 6
            point_cloud = point_cloud.transpose(1, 2)  # 转置为 B x 6 x N
            shape_code = shape_encoder(point_cloud)  # B x shape_feature_dim
            
            query_pos = query_points[:,:3] # B x 3
            query_pos = query_pos.unsqueeze(2) # B x 3 x 1
            query_code = pos_encoder(query_pos) # B x pos_feature_dim
            
            qs_feature = torch.cat([shape_code, query_code],dim=-1) # B x (shape_feature_dim + pos_feature_dim)
            
            sdf_pred = sdf_decoder(qs_feature) # B x 1
            gt_y = sdistane
            
            # Compute loss
            loss = nn.MSELoss()(sdf_pred.squeeze(), gt_y.squeeze())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % config["training"]["log_interval"] == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
                
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
    # Set data type
    torch_dtype = get_torch_dtype(config.get("dtype", {}).get("torch", "float32"))
    
    # Initialize model components
    shape_feature_dim = config["model"]["shape_encoder"]["output_dim"]
    pos_feature_dim = config["model"]["pos_encoder"]["output_dim"]
    shape_encoder = ShapeFeatureExtractor(input_channels=6, output_dim=shape_feature_dim)
    pos_encoder = ShapeFeatureExtractor(input_channels=3, output_dim=pos_feature_dim)
    sdf_decoder = SDFDecoder(input_dim=shape_feature_dim + pos_feature_dim)
    
    # Move to device
    device = torch.device(config["device"]["cuda"])
    shape_encoder.to(device)
    pos_encoder.to(device)
    sdf_decoder.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    shape_encoder.load_state_dict(checkpoint['shape_encoder_state_dict'])
    pos_encoder.load_state_dict(checkpoint['pos_encoder_state_dict'])
    sdf_decoder.load_state_dict(checkpoint['sdf_decoder_state_dict'])
    
    # Set models to evaluation mode
    shape_encoder.eval()
    pos_encoder.eval()
    sdf_decoder.eval()
    
    # Load dataset
    point_cloud_dataset = PointCloudDataset(config["test"]["point_cloud_path"])
    query_dataset = QueryPointsDataset(
        pointcloudDataset=point_cloud_dataset,
        resolution=config["field"]["resolution"],
        k=config["query"]["k_neighbors"],
        device=device,
        save_path=config["data"]["save_path"]
    )
    
    all_points = query_dataset.point_list
    all_points = np.asarray(all_points)
    all_points = torch.from_numpy(all_points).to(device)
    
    all_normals = torch.zeros_like(all_points)
    for i in range(all_points.shape[0]):
        tpoint = all_points[i].cpu().numpy()
        tnormal = tools.pymeshlab_normal_estimate(tpoint, 10)
        tnormal = torch.from_numpy(tnormal).to(device)
        all_normals[i] = tnormal
    
    # Create output directory
    output_dir = config["test"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each point cloud model
    for model_idx in range(len(all_points)):
        print(f"Processing model {model_idx+1}/{len(all_points)}")
        
        # Get current point cloud and normals
        points = all_points[model_idx].cpu().numpy()
        normals = all_normals[model_idx].cpu().numpy()
        
        # Create input for shape encoder
        oriented_points = np.concatenate([points, normals], axis=-1)
        oriented_points = torch.from_numpy(oriented_points).to(device)
        oriented_points = oriented_points.unsqueeze(0) # 1 x N x 6
        oriented_points = oriented_points.transpose(1, 2)  # 转置为 1 x 6 x N
        
        
        # Generate shape code
        with torch.no_grad():
            shape_code = shape_encoder(oriented_points)  # 1 x shape_feature_dim
        
        # Create a grid of query points
        resolution = config["test"]["resolution"]
        bbox = np.array(config["field"]["bbox"])
        query_points, grid_shape = tools.create_uniform_grid(resolution=resolution, bbox=bbox)
        query_points_torch = torch.tensor(query_points, dtype=torch_dtype, device=device)
        
        # Compute SDF values for each query point
        sdf_values = []
        batch_size = config["test"]["batch_size"]
        num_batches = (query_points.shape[0] + batch_size - 1) // batch_size
        
        print(f"Computing SDF values for model {model_idx+1}...")
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, query_points.shape[0])
                
                batch_query_points = query_points_torch[start_idx:end_idx].unsqueeze(2)  # B x 3 x 1
                batch_query_code = pos_encoder(batch_query_points)  # B x pos_feature_dim
                
                # Repeat shape code for each query point
                batch_shape_code = shape_code.repeat(batch_query_code.shape[0], 1)  # B x shape_feature_dim
                
                # Concatenate shape and position features
                batch_features = torch.cat([batch_shape_code, batch_query_code], dim=1)  # B x (shape_feature_dim + pos_feature_dim)
                
                # Predict SDF values
                batch_sdf = sdf_decoder(batch_features).cpu().numpy()  # B x 1
                sdf_values.append(batch_sdf)
        
        # Combine all SDF values and reshape to 3D grid
        sdf_values = np.concatenate(sdf_values, axis=0).reshape(grid_shape)
        sdf_field = SDFField(resolution=resolution)
        gt_sdf_values = sdf_field.compute_sdf(all_points[model_idx].cpu(), all_normals[model_idx].cpu(), query_points)
        gt_sdf_values = gt_sdf_values.cpu().numpy()
        loss = np.linalg.norm(sdf_values.reshape(-1) - gt_sdf_values.reshape(-1))
        print(f"Loss: {loss}")
        
        
        # Extract isosurface
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
        
        # Compute normals for the original point cloud using the extracted mesh
        print(f"Computing normals from extracted mesh for model {model_idx+1}...")
        updated_normals = compute_point_normals_from_mesh(points, verts, faces, k=10)
        
        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, f"model_{model_idx}")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Save mesh
        mesh_output_path = os.path.join(model_output_dir, "reconstructed_mesh.ply")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(mesh_output_path, mesh)
        print(f"Mesh saved to {mesh_output_path}")
        
        # Save point cloud with updated normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(updated_normals)
        pcd_output_path = os.path.join(model_output_dir, "point_cloud_with_normals.ply")
        o3d.io.write_point_cloud(pcd_output_path, pcd)
        print(f"Point cloud with updated normals saved to {pcd_output_path}")
        
        # Optionally, visualize the results
        tools.plot_partition(
            sdf_values > 0, 
            save_path=os.path.join(model_output_dir, "sdf_slice.png"),
            slice_idx=None,
            bbox=bbox,
            points=points,
            vis=False
        )
        
        # Save voxel grid
        voxel_output_dir = config["test"]["voxel_output_dir"]
        if voxel_output_dir is not None:
            os.makedirs(voxel_output_dir, exist_ok=True)
            voxel_output_path = os.path.join(voxel_output_dir, f"voxel_grid_{model_idx}.ply")
            # 保存sdf_values
            np.save(voxel_output_path, sdf_values)
            print(f"Voxel grid saved to {voxel_output_path}")
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
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
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