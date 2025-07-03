import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from data_util.surface_sampling_dataloader import get_surface_sampling_dataloader
from pipline.config import get_config # Import config
from field import MeshSDF,SDFField
import tools
import models.trellis.modules.sparse as sp
import models.lzd_models.slat_net as lat_net
from models.lzd_models.slat_net import SparseResConv3d
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
import torch.nn.functional as F


# class GridEncoder(nn.Module):
#     def __init__(self):
#         super(GridEncoder, self).__init__()
#         encoder_cfg = json.load(open("/mnt/lizd/work/CG/NormalEstimation/iSDF/config/encoder_config.json")).get("model")
#         config1 = encoder_cfg.get("SparseConv3dEncoder").get("args")
#         self.ss_encoder = SparseStructureEncoder(**config1)

#     def forward(self, x: torch.Tensor):
#         x = self.ss_encoder(x)
#         return x

class SLatEncoder(nn.Module):
    def __init__(self,in_dim: int):
        super(SLatEncoder, self).__init__()
        encoder_cfg = json.load(open("/mnt/lizd/work/CG/NormalEstimation/iSDF/config/encoder_config.json")).get("model").get("SparseLatentEncoder").get("args")
        self.input_dim = in_dim
        self.input_transform = nn.Sequential(
            lat_net.L_L_T(self.input_dim, encoder_cfg.get("in_channels")),
            lat_net.L_L_T(encoder_cfg.get("in_channels"), encoder_cfg.get("in_channels")),
        )
        # TODO: 自制一个encoder（vae的encoder会加噪）
        self.slat_encoder = lat_net.SLatEncoder(**encoder_cfg)

    def forward(self, x: sp.SparseTensor):
        x = self.input_transform(x)
        return self.slat_encoder(x)


class GridDecoder(nn.Module):
    def __init__(self):
        super(GridDecoder, self).__init__()
    
        ss_decoder_cfg = json.load(open("/mnt/lizd/work/CG/NormalEstimation/iSDF/config/decoder_config.json")).get("model").get("SlatVoxelDecoder").get("args")
        
        self.ss_decoder = lat_net.SLatVoxelDecoder(**ss_decoder_cfg)
        self.activation = nn.Tanh()

        
    def forward(self, x):
        x = self.ss_decoder(x)
        feats = self.activation(x.feats)
        x = x.replace(feats)
        return x


class VoxelGridVAEBuilder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(VoxelGridVAEBuilder, self).__init__()
        config = json.load(open("/mnt/lizd/work/CG/NormalEstimation/iSDF/config/voxelGridVAE.json")).get("model")    
        encoder_config = config.get("SparseLatentEncoder").get("args")
        decoder_config = config.get("SlatVoxelDecoder").get("args")
    
        encoder_config["in_channels"] = config.get("io_block_channels")[-1]
        encoder_config["use_fp16"] = config.get("use_fp16")
        decoder_config["use_fp16"] = config.get("use_fp16")
        
        self.model = lat_net.VoxelGridVAE(
            input_channels=input_channels,
            output_channels=output_channels,
            num_io_res_blocks=config.get("num_io_res_blocks"),
            io_block_channels=config.get("io_block_channels"),
            use_fp16=config.get("use_fp16"),
            model_channels=config.get("model_channels"),
            use_skip_connections=config.get("use_skip_connections"),
            encoder_config=encoder_config,
            decoder_config=decoder_config
        )
        
    def forward(self, x):
        return self.model(x)

# --- Placeholder Helper Functions ---
def compute_gt_sdf_grid(vertices, faces, R, device,masks):
    """
    Placeholder: Computes a dense R x R x R SDF grid from mesh.
    This function would typically use a library like trimesh.
    Input:
        vertices: Batch of vertices (B, N_verts, 3)
        faces: Batch of faces (B, N_faces, 3)
        R: Resolution of the SDF grid
    Output:
        gt_sdf: Batch of SDF grids (B, R, R, R)
    """
    # Example: return a dummy grid for now
    # In a real implementation, you'd iterate through the batch,
    # create a trimesh.Trimesh object for each, then use its SDF querying capabilities.
    # This requires defining a bounding box for the grid.

    batch_size = len(vertices)
    gt_sdf = torch.zeros(batch_size, R, R, R, device=device)
    query_points,grid_shape = tools.create_uniform_grid(R, bbox=np.array([[-1,1],[-1,1],[-1,1]]))

    for i in range(batch_size):
        mesh_sdf = MeshSDF(vertices[i].cpu().numpy(), faces[i].cpu().numpy())
        mask = masks[i].cpu().numpy()
        query_points = query_points.reshape(grid_shape)[mask]
        query_points = query_points.reshape(-1,3)
        sdf_values = mesh_sdf.query(query_points)
        sdf_values = sdf_values.reshape(grid_shape)
        gt_sdf[i] = torch.from_numpy(sdf_values).to(device)
    return gt_sdf

def estimate_normals_pca(points, k):
    """
    Placeholder: Estimates normals for points using PCA on k-nearest neighbors.
    Input:
        points: (B, N_points, 3)
        k: number of neighbors for PCA
    Output:
        pred_normals: (B, N_points, 3)
    """
    # This would involve:
    # 1. For each point, find k-nearest neighbors.
    # 2. Compute covariance matrix for the neighborhood.
    # 3. Eigen decomposition to find the normal (eigenvector corresponding to smallest eigenvalue).
    # 4. Ensure consistent orientation (e.g., towards a viewpoint or average direction).
    
    batch_size = points.shape[0]
    pred_normals = torch.zeros(batch_size, points.shape[1], 3, device=points.device)
    tpoints = points.cpu().numpy()
    for i in range(batch_size):
        pred_normals[i] = torch.from_numpy(tools.pymeshlab_normal_estimate(tpoints[i], k)).to(points.device)
    return pred_normals

def op2wnf(points, pred_normals, r, device):
    """
    Placeholder: Creates an r x r x r voxel grid based on points and predicted normals.
    The content of this grid needs to be defined (e.g., occupancy, feature based on normals).
    Input:
        points: (B, N_points, 3)
        pred_normals: (B, N_points, 3)
        r: resolution of the voxel grid
    Output:
        voxel_grid_features: (B, r, r, r) or (B, C, r, r, r)
                           This should be convertible to spconv.SparseConvTensor
    """
    # This is a complex step. How normals define the grid content is key.
    # For now, returning a dummy occupancy grid based on point locations.
    # A real implementation would discretize points/normals into a grid.
    import cal_wnf
    batch_size = points.shape[0]
    voxel_grid = torch.zeros(batch_size, r, r, r, device=device)
    query_points,grid_shape = tools.create_uniform_grid(r, bbox=np.array([[-1,1],[-1,1],[-1,1]]))
    masks = torch.zeros(batch_size, r, r, r, device=device,dtype=torch.bool)
    query_points = torch.from_numpy(query_points).to(device)
    
    for i in range(batch_size):
        wnf = cal_wnf.compute_winding_number_torch_api(points[i], pred_normals[i],query_points,epsilon=1e-8,batch_size=10000)
        mask = tools.create_mask_by_k(query_points.cpu().numpy(),points[i].cpu().numpy(),k=60)
        mask = torch.from_numpy(mask).to(device)
        mask = mask.reshape(grid_shape)
        wnf = wnf.reshape(grid_shape)
        masks[i] = mask
        voxel_grid[i] = wnf
    return voxel_grid,masks # Shape (B, r, r, r)

# 自定义loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
    def forward(self, pred, target):
        loss = 1 - torch.abs(torch.mean(pred*target))
        return loss * 1000



def check_gradients():
    """检查梯度是否正常"""
    print("\n=== Gradient Check ===")
    
    for name, model in [vae]:
        print(f"\n{name} gradients:")
        total_norm = 0
        param_count = 0
        
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                if torch.isnan(param.grad).any():
                    print(f"  ❌ {param_name}: NaN gradient")
                elif torch.isinf(param.grad).any():
                    print(f"  ❌ {param_name}: Inf gradient")
                elif param_norm > 10.0:  # 检查梯度爆炸
                    print(f"  ⚠️ {param_name}: Large gradient norm {param_norm:.4f}")
                else:
                    print(f"  ✅ {param_name}: norm {param_norm:.4f}")
        
        total_norm = total_norm ** (1. / 2)
        print(f"  Total gradient norm: {total_norm:.4f}")

def get_total_grad_norm(model):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** (1. / 2)

def check_model_params(model, model_name="Model"):
    """检查模型参数中的非法值"""
    has_nan = False
    has_inf = False
    
    print(f"\n=== Checking {model_name} Parameters ===")
    
    for name, param in model.named_parameters():
        # 检查 NaN
        if torch.isnan(param).any():
            print(f"❌ NaN found in {name}")
            has_nan = True
        
        # 检查 inf
        if torch.isinf(param).any():
            print(f"❌ Inf found in {name}")
            has_inf = True
        
        # 检查梯度
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"❌ NaN found in gradient of {name}")
                has_nan = True
            if torch.isinf(param.grad).any():
                print(f"❌ Inf found in gradient of {name}")
                has_inf = True
    
    if not has_nan and not has_inf:
        print(f"✅ {model_name} parameters are healthy")
    
    return has_nan, has_inf

def evaluate_model(model, dataloader, device, cfg, loss_fn, epoch=None, save_results=False):
    """
    Evaluates the model on the validation/test set using **exactly** the same
    data-preparation pipeline as the training loop so that no distribution
    shift is introduced between training and evaluation.
    """
    vae.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_acc_wnf = 0.0
    total_feats_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (points, gt_normals, vertices, faces) in enumerate(dataloader):
            points = points.to(device)  # (B, N_pts, 3)
            B = points.size(0)

            # 1. GT SDF grid
            gt_sdf = compute_gt_sdf_grid(vertices, faces, cfg["R"], device)  # (B, R, R, R)
            gt_sdf = gt_sdf.unsqueeze(1) 
            gt_udf = abs(gt_sdf)# (B, 1, R, R, R)
            gt_sdf[gt_sdf > 0.0] = 1
            gt_sdf[gt_sdf < 0.0] = -1
            

            # 2. Estimate normals with PCA
            pred_normals = estimate_normals_pca(points, k=cfg["pca_knn"])

            # 3. Create voxel grid + mask from predicted normals
            voxel_grid_features, masks = op2wnf(points, pred_normals, cfg["r"], device)  # (B, r, r, r)
            gt_wnf, _ = op2wnf(points, gt_normals, cfg["r"], device)

            # 4. UDF on the masked voxels (keeps consistency with training)
            sdf_field = SDFField(cfg["r"])
            udf = torch.zeros([B, cfg["r"], cfg["r"], cfg["r"]], device=device)
            for i in range(B):
                qp, ssq = tools.create_uniform_grid(cfg["r"], bbox=np.array([[-1, 1], [-1, 1], [-1, 1]]))
                qp = qp.reshape(ssq[0], ssq[1], ssq[2], 3)
                qp = qp[masks[i].cpu().numpy()]
                udf[i][masks[i]] = torch.abs(sdf_field.compute_sdf(points[i], pred_normals[i], qp))
            udf = udf.unsqueeze(1)                                             # (B, 1, r, r, r)
            # udf = torch.tanh(udf)

            # 5. Gather sparse voxels (mask == True)
            indices = torch.nonzero(masks).int()  # (M, 4) -> (batch, z, y, x)

            gt_wnf_val = gt_wnf[indices[:, 0], None, indices[:, 1], indices[:, 2], indices[:, 3]]
            gt_wnf_val = torch.tanh(gt_wnf_val)  # (M, 1)
            udf_val = udf[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]  # (M, 1)
            gt_udf_val = gt_udf[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]  # (M, 1)

            feats = voxel_grid_features[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]].unsqueeze(1)  # (M, 1)
            feats = torch.tanh(feats)
            feats_acc = (feats > 0.0).eq(gt_wnf_val > 0.0).float().mean()

            feats[feats > 0.0] = 1.0
            feats[feats < 0.0] = -1.0
            feats = torch.cat([feats,udf_val], dim=1)  # (M, 2)

            # 6. Forward pass
            sp_feats = sp.SparseTensor(feats, indices)
            sp_feats = vae(sp_feats)
            
            pred_val = sp_feats.feats
            indices_out = sp_feats.coords
            gt_sdf_val = gt_sdf[indices_out[:, 0], :, indices_out[:, 1], indices_out[:, 2], indices_out[:, 3]]

            # 7. Metrics
            loss = loss_fn(pred_val, gt_wnf_val)
            acc = (pred_val > 0.0).eq(gt_wnf_val > 0.0).float().mean()
            acc_wnf = (gt_sdf_val > 0.0).eq(gt_wnf_val > 0.0).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            total_acc_wnf += acc_wnf.item()
            total_feats_acc += feats_acc.item()
            num_batches += 1
            
            # 8. Save validation results if requested
            if save_results and epoch is not None:
                val_save_dir = f"val_results/epoch_{epoch}"
                os.makedirs(val_save_dir, exist_ok=True)
                
                # For each item in the batch
                for i in range(B):
                    batch_save_dir = f"{val_save_dir}/batch_{batch_idx}_item_{i}"
                    os.makedirs(batch_save_dir, exist_ok=True)
                    
                    # Create full prediction SDF grid
                    pred_sdf = torch.zeros_like(gt_sdf)
                    batch_mask = indices_out[:, 0] == i
                    if batch_mask.any():
                        batch_indices = indices_out[batch_mask]
                        batch_pred_val = pred_val[batch_mask]
                        pred_sdf[i, :, batch_indices[:, 1], batch_indices[:, 2], batch_indices[:, 3]] = batch_pred_val
                    
                    # Save meshes
                    try:
                        # Extract and save GT SDF surface
                        tools.extract_surface_from_scalar_field(
                            gt_sdf[i].squeeze().cpu().numpy(), 
                            level=0, 
                            resolution=cfg["R"], 
                            save_path=f"{batch_save_dir}/gt_sdf.ply",
                            mask=None  # Use full grid for GT
                        )
                        
                        # Extract and save predicted SDF surface
                        tools.extract_surface_from_scalar_field(
                            pred_sdf[i].squeeze().cpu().detach().numpy(), 
                            level=0, 
                            resolution=cfg["R"], 
                            save_path=f"{batch_save_dir}/pred_sdf.ply",
                            mask=masks[i].cpu().numpy()
                        )
                        
                        # Extract and save GT WNF surface
                        tools.extract_surface_from_scalar_field(
                            gt_wnf[i].squeeze().cpu().numpy(), 
                            level=0, 
                            resolution=cfg["r"], 
                            save_path=f"{batch_save_dir}/gt_wnf.ply",
                            mask=masks[i].cpu().numpy()
                        )
                        
                        # Save original GT mesh
                        import open3d as o3d
                        o3d.io.write_triangle_mesh(
                            f"{batch_save_dir}/gt_mesh.ply",
                            o3d.geometry.TriangleMesh(
                                o3d.utility.Vector3dVector(vertices[i].cpu().numpy()),
                                o3d.utility.Vector3iVector(faces[i].cpu().numpy())
                            )
                        )
                        
                        # Save point cloud with normals
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points[i].cpu().numpy())
                        pcd.normals = o3d.utility.Vector3dVector(pred_normals[i].cpu().numpy())
                        o3d.io.write_point_cloud(f"{batch_save_dir}/input_points_with_normals.ply", pcd)
                        
                    except Exception as e:
                        print(f"Warning: Could not save mesh for validation batch {batch_idx}, item {i}: {e}")
                    
                    # Save scalar fields as numpy arrays
                    try:
                        np.save(f"{batch_save_dir}/gt_sdf.npy", gt_sdf[i].cpu().numpy())
                        np.save(f"{batch_save_dir}/pred_sdf.npy", pred_sdf[i].cpu().detach().numpy())
                        np.save(f"{batch_save_dir}/gt_wnf.npy", gt_wnf[i].cpu().numpy())
                        np.save(f"{batch_save_dir}/udf.npy", udf[i].cpu().numpy())
                        np.save(f"{batch_save_dir}/mask.npy", masks[i].cpu().numpy())
                        np.save(f"{batch_save_dir}/voxel_grid_features.npy", voxel_grid_features[i].cpu().numpy())
                        
                        # Save input data
                        np.save(f"{batch_save_dir}/input_points.npy", points[i].cpu().numpy())
                        np.save(f"{batch_save_dir}/pred_normals.npy", pred_normals[i].cpu().numpy())
                        np.save(f"{batch_save_dir}/gt_normals.npy", gt_normals[i].cpu().numpy())
                        np.save(f"{batch_save_dir}/vertices.npy", vertices[i].cpu().numpy())
                        np.save(f"{batch_save_dir}/faces.npy", faces[i].cpu().numpy())
                        
                    except Exception as e:
                        print(f"Warning: Could not save numpy arrays for validation batch {batch_idx}, item {i}: {e}")
                    
                    # Save metrics for this sample
                    try:
                        metrics = {
                            'loss': loss.item(),
                            'accuracy': acc.item(),
                            'accuracy_wnf': acc_wnf.item(),
                            'feats_accuracy': feats_acc.item(),
                            'num_valid_voxels': masks[i].sum().item(),
                            'total_voxels': masks[i].numel()
                        }
                        
                        with open(f"{batch_save_dir}/metrics.json", 'w') as f:
                            json.dump(metrics, f, indent=2)
                            
                    except Exception as e:
                        print(f"Warning: Could not save metrics for validation batch {batch_idx}, item {i}: {e}")
                
                # Only save first few batches to avoid too much storage
                if batch_idx >= 2:  # Save first 3 batches
                    break

    # 9. Aggregate statistics
    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    avg_acc_wnf = total_acc_wnf / max(num_batches, 1)
    avg_feats_acc = total_feats_acc / max(num_batches, 1)

    return avg_loss, avg_acc, avg_acc_wnf, avg_feats_acc

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network for SDF estimation')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (e.g., 0, 1, etc.)')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation set split ratio')
    parser.add_argument('--save_val_results', action='store_true', help='Save validation results (meshes, grids, etc.)')
    
    args = parser.parse_args()

    # Load configuration
    cfg = get_config(dataset_name=args.dataset_name)
    
    # Set device based on command-line argument
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    
    print(f"Using device: {device}")

    # Initialize Dataloaders with validation split
    train_dataloader, val_dataloader = get_surface_sampling_dataloader(
        data_path=cfg["data_path"],
        num_samples=cfg["num_samples"],
        batch_size=cfg["batch_size"],
        save_path=cfg["save_path"],
        use_torch=cfg["use_torch_dataloader"],
        device=device,
        normalize=True,
        shuffle=True,
        val_split=args.val_split
    )
    
    print(f"Training set size: {len(train_dataloader.dataset)}")
    print(f"Validation set size: {len(val_dataloader.dataset)}")

    # Initialize Models
    # slat_encoder = SLatEncoder(in_dim=2).to(device)
    # grid_decoder = GridDecoder().to(device)
    vae = VoxelGridVAEBuilder(input_channels=2,output_channels=1).to(device)    


    # Optimizer
    params_to_optimize = (
        list(vae.parameters())
    )
    optimizer = optim.Adam(params_to_optimize, lr=cfg["learning_rate"], eps=1e-4)

    # Loss Function
    loss_fn = CustomLoss()
    
    # Initialize TensorBoard SummaryWriter
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/train_{args.dataset_name}_{timestamp}" if args.dataset_name else f"runs/train_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={log_dir}")
    
    # Log hyperparameters
    hparams = {
        'learning_rate': cfg["learning_rate"],
        'batch_size': cfg["batch_size"],
        'epochs': cfg["epochs"],
        'R': cfg["R"],
        'r': cfg["r"],
        'pca_knn': cfg["pca_knn"],
        'num_samples': cfg["num_samples"],
        'val_split': args.val_split,
        'device': str(device)
    }
    writer.add_hparams(hparams, {'hparam/best_val_loss': float('inf'), 'hparam/best_val_acc': 0.0})
    
    # Training Loop
    pointcloud_voxel_resolution = cfg["r"]
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # Add model architecture to tensorboard (will be done on first forward pass)
    model_logged = False
    
    for epoch in range(cfg["epochs"]):
        # Training phase
        vae.train()
        
        for batch_idx, (points, gt_normals, vertices, faces) in enumerate(train_dataloader):
            # Ensure data is on the correct device
            points = points.to(device)         # (B, N_points, 3)

            # 1. Encode points with PointNet -> f1
            voxel_coords = points.clone()
            assert points.max() <= 1.0 and points.min() >= -1.0
            voxel_coords = (voxel_coords+1.0)/2.0
            voxel_coords = (voxel_coords*pointcloud_voxel_resolution).long()

            B, N, _ = voxel_coords.shape
            batch_indices = torch.arange(B, device=voxel_coords.device).repeat_interleave(N).unsqueeze(-1)  # [N*B, 1]
            voxel_coords = torch.cat([batch_indices, voxel_coords.reshape(-1, 3)], dim=-1).int()  # [N*B, 4]
            vae.train()    # per_point_feats = pointnet_encoder(points,voxel_coords,res=pointcloud_voxel_resolution)
            # pc_sparse_slat_feats = sp.SparseTensor(per_point_feats,voxel_coords)
            # pc_sparse_slat_feats = torch.zeros(B,1,pointcloud_voxel_resolution,pointcloud_voxel_resolution,pointcloud_voxel_resolution,device=device)

            pred_normals = estimate_normals_pca(points, k=cfg["pca_knn"]) # (B, N_points, 3)

            i_resolu = cfg["r"]
            o_resolu = cfg["R"]
            voxel_center,grid_shape = tools.create_uniform_grid(i_resolu,bbox=np.array([[-1,1],[-1,1],[-1,1]]))
            feat_shape = [B] + list(grid_shape)
            gt_wnf = torch.zeros(feat_shape,device=device)
            pred_wnf = torch.zeros(feat_shape,device=device)
            gt_sdf = torch.zeros(feat_shape,device=device)
            gt_udf = torch.zeros(feat_shape,device=device)
            pred_udf = torch.zeros(feat_shape,device=device)
            masks = torch.zeros(feat_shape,device=device)
            
            import cal_wnf
            sdf_field = SDFField(i_resolu)
            for i in range(B):
                mask = tools.create_mask_by_k(voxel_center,points[i].cpu().numpy(),k=60)    
                query_points_i = voxel_center[mask]
                mask_cuda = torch.from_numpy(mask).to(device).reshape(grid_shape)
                masks[i] = mask_cuda
                mask = mask.reshape(grid_shape)
                query_points_i = torch.from_numpy(query_points_i).to(device)
                pred_normals_i = pred_normals[i]
                
                # 计算wnf
                wnf_i = cal_wnf.compute_winding_number_torch_api(points[i],pred_normals[i],query_points_i,epsilon=1e-8,batch_size=10000)
                gt_wnf_i = cal_wnf.compute_winding_number_torch_api(points[i],gt_normals[i],query_points_i,epsilon=1e-8,batch_size=10000)
                pred_wnf[i][mask_cuda] = wnf_i
                gt_wnf[i][mask_cuda] = gt_wnf_i
                
                # 计算sdf
                mesh_sdf = MeshSDF(vertices[i].cpu().numpy(), faces[i].cpu().numpy())
                sdf_values = mesh_sdf.query(query_points_i.cpu().numpy())
                gt_sdf[i][mask_cuda] = torch.from_numpy(sdf_values).to(device,dtype=torch.float32)
                
                # 计算udf
                gt_udf[i][mask_cuda] = torch.abs(gt_sdf[i][mask_cuda])
                pred_udf[i][mask_cuda] = torch.abs(sdf_field.compute_sdf(points[i],pred_normals[i],query_points_i))
                
                
            indices = torch.nonzero(masks).int()
            gt_sdf_val = gt_sdf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            gt_udf_val = gt_udf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            pred_udf_val = pred_udf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            wnf_val = pred_wnf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            gt_wnf_val = gt_wnf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            
            feats1 = wnf_val
            feats1 = torch.tanh(feats1)
            feats_acc = (feats1>0.0).squeeze().eq(gt_wnf_val>0.0).float().mean()
            feats12 = torch.cat([feats1.unsqueeze(1),pred_udf_val.unsqueeze(1)],dim=1)
            
            sp_feats = sp.SparseTensor(feats12, indices)
            sp_feats = vae(sp_feats)
            pred_val = sp_feats.feats
            
                
            

            # voxel_grid_features,masks = op2wnf(points, pred_normals, cfg["r"], device) # (B, r, r, r)
            # gt_wnf,_ = op2wnf(points, gt_normals, cfg["r"], device) # (B, r, r, r)
            

            # # 2. Compute GT SDF grid (R x R x R)
            # gt_sdf = compute_gt_sdf_grid(vertices, faces, cfg["R"], device,masks) # (B, R, R, R) 
            # gt_sdf = gt_sdf.unsqueeze(1)
            # gt_udf = abs(gt_sdf)
            # gt_sdf[gt_sdf>0.0] = 1
            # gt_sdf[gt_sdf<0.0] = -1
            #             # 计算UDF
            # sdf_field = SDFField(cfg["r"])
            # udf = torch.zeros([B,cfg["r"],cfg["r"],cfg["r"]]).to(device)
            # for i in range(B):
            #     qp,ssq = tools.create_uniform_grid(cfg["r"],bbox=np.array([[-1,1],[-1,1],[-1,1]]))
            #     qp = qp.reshape(ssq[0],ssq[1],ssq[2],3)
            #     qp = qp[masks[i].cpu().numpy()]
            #     udf[i][masks[i]] = torch.abs(sdf_field.compute_sdf(points[i],pred_normals[i],qp))
            # udf = udf.unsqueeze(1)
            # # udf = torch.tanh(udf)
            
            # indices = torch.nonzero(masks).int()            
            # gt_wnf_val = gt_wnf[indices[:,0],None,indices[:,1],indices[:,2],indices[:,3]]
            # gt_wnf_val = torch.tanh(gt_wnf_val) 
            # udf_val = udf[indices[:,0],:,indices[:,1],indices[:,2],indices[:,3]]
            # gt_udf_val = gt_udf[indices[:,0],:,indices[:,1],indices[:,2],indices[:,3]]
            # feats1 = voxel_grid_features[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            # feats1 = feats1.unsqueeze(1)
            # feats1 = torch.tanh(feats1)
            # feats_acc = (feats1>0.0).eq(gt_wnf_val>0.0).float().mean()
            
            # assert not voxel_grid_features.isnan().any()
            
            # feats1[feats1>0.0] = 1.0
            # feats1[feats1<0.0] = -1.0
            
            # feats12 = torch.cat([feats1,udf_val],dim=1)
            # gt_sdf_val = gt_sdf[indices[:,0],:,indices[:,1],indices[:,2],indices[:,3]]
            
            # sp_feats = sp.SparseTensor(feats12, indices)
            # sp_feats = vae(sp_feats)
            # batch_size = points.shape[0]
            # indices = sp_feats.coords
            # gt_sdf_val = gt_sdf[indices[:,0],:,indices[:,1],indices[:,2],indices[:,3]]
            # pred_val = sp_feats.feats
            
            # Log model graph to tensorboard (only once)
            if not model_logged and batch_idx == 0 and epoch == 0:
                try:
                    # Create a sample input for graph logging
                    sample_input = sp.SparseTensor(feats12[:100], indices[:100])  # Use first 100 points
                    writer.add_graph(vae, sample_input)
                    model_logged = True
                    print("Model graph added to TensorBoard")
                except Exception as e:
                    print(f"Could not add model graph to TensorBoard: {e}")
                    model_logged = True  # Don't try again
            
            loss = loss_fn(pred_val, gt_wnf_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_grad_norm_encoder = get_total_grad_norm(vae.encoder)
            total_grad_norm_decoder = get_total_grad_norm(vae.decoder)
            
            # Calculate global step for tensorboard logging
            global_step = epoch * len(train_dataloader) + batch_idx
            
            # Log to tensorboard every batch
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/GradNorm_Encoder', total_grad_norm_encoder, global_step)
            writer.add_scalar('Train/GradNorm_Decoder', total_grad_norm_decoder, global_step)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            print(f"Epoch {epoch+1}/{cfg['epochs']}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.6f}, encoder_Total_grad_norm: {total_grad_norm_encoder:.6f}, decoder_Total_grad_norm: {total_grad_norm_decoder:.6f}")

            if batch_idx % cfg["log_interval"] == 0:
                acc = (pred_val>0.0).eq(gt_wnf_val>0.0).float().mean()
                loss_gt_wnf = loss_fn(gt_wnf_val,gt_sdf_val)
                acc_wnf = (gt_sdf_val>0.0).eq(gt_wnf_val>0.0).float().mean()
                
                # Log detailed metrics to tensorboard
                writer.add_scalar('Train/Accuracy', acc.item(), global_step)
                writer.add_scalar('Train/Accuracy_WNF', acc_wnf.item(), global_step)
                writer.add_scalar('Train/Feats_Accuracy', feats_acc.item(), global_step)
                writer.add_scalar('Train/Loss_GT_WNF', loss_gt_wnf.item(), global_step)
                
                print(f"Epoch {epoch+1}/{cfg['epochs']}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.6f}, Acc: {acc.item():.6f}, Acc_wnf: {acc_wnf.item():.6f}, Loss_gt_wnf: {loss_gt_wnf.item():.6f}, Feats_acc: {feats_acc.item():.6f}")
                ifview = False
                if True:
                    os.makedirs("train_results",exist_ok=True)
                    os.makedirs("train_results/mesh_{}".format(batch_idx),exist_ok=True)
                    tools.extract_surface_from_scalar_field(gt_sdf[0].squeeze().cpu().numpy(),level=0,resolution=cfg["r"],save_path="train_results/mesh_{}/gt_sdf.ply".format(batch_idx),mask=mask_cuda[0].cpu().numpy())
                    pred_sdf = torch.zeros_like(gt_sdf)
                    pred_sdf[indices[:,0],:,indices[:,1],indices[:,2],indices[:,3]] = pred_val
                    tools.extract_surface_from_scalar_field(pred_sdf[0].squeeze().cpu().detach().numpy(),level=0,resolution=cfg["r"],save_path="train_results/mesh_{}/pred_sdf.ply".format(batch_idx),mask=mask_cuda[0].cpu().numpy())
                    tools.extract_surface_from_scalar_field(gt_wnf[0].squeeze().cpu().numpy(),level=0,resolution=cfg["r"],save_path="train_results/mesh_{}/gt_wnf.ply".format(batch_idx),mask=mask_cuda[0].cpu().numpy())
                    # 保存vertices和faces为ply
                    import open3d as o3d
                    o3d.io.write_triangle_mesh("train_results/mesh_{}/gt_mesh.ply".format(batch_idx),o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices[0].cpu().numpy()),o3d.utility.Vector3iVector(faces[0].cpu().numpy())))
                    # 保存field
                    np.save("train_results/mesh_{}/gt_sdf.npy".format(batch_idx),gt_sdf[0].cpu().numpy())
                    np.save("train_results/mesh_{}/pred_sdf.npy".format(batch_idx),pred_sdf[0].cpu().detach().numpy())
                    np.save("train_results/mesh_{}/gt_wnf.npy".format(batch_idx),gt_wnf[0].cpu().numpy())
                    
                    # Log some scalar field slices to tensorboard for visualization
                    if batch_idx == 0:  # Only log for first batch to avoid cluttering
                        # Log middle slice of the 3D fields
                        mid_slice = cfg["r"] // 2
                        writer.add_image('Visualization/GT_SDF_Slice', 
                                       gt_sdf[0, 0, mid_slice:mid_slice+1, :, :], 
                                       global_step, dataformats='CHW')
                        writer.add_image('Visualization/Pred_SDF_Slice', 
                                       pred_sdf[0, 0, mid_slice:mid_slice+1, :, :], 
                                       global_step, dataformats='CHW')
                        writer.add_image('Visualization/GT_WNF_Slice', 
                                       gt_wnf[0, None, mid_slice, :, :], 
                                       global_step, dataformats='CHW')
                        writer.add_image('Visualization/GT_UDF_Slice', 
                                       gt_udf[0, 0, mid_slice:mid_slice+1, :, :], 
                                       global_step, dataformats='CHW')
                    # import view

        # Validation phase
        save_val_results = args.save_val_results and (epoch % 5 == 0 or epoch == cfg["epochs"] - 1)  # Save every 5 epochs and last epoch
        val_loss, val_acc, val_acc_wnf, val_feats_acc = evaluate_model(
            vae, val_dataloader, device, cfg, loss_fn, epoch, save_val_results
        )
        
        # Log validation metrics to tensorboard
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        writer.add_scalar('Val/Accuracy_WNF', val_acc_wnf, epoch)
        writer.add_scalar('Val/Feats_Accuracy', val_feats_acc, epoch)
        
        print(f"\nValidation Results - Epoch {epoch+1}:")
        print(f"Loss: {val_loss:.6f}, Acc: {val_acc:.6f}")
        print(f"Acc_wnf: {val_acc_wnf:.6f}, Feats_acc: {val_feats_acc:.6f}")
        
        # Regular checkpoint saving
        os.makedirs("temp/checkpoints",exist_ok=True)
        
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, "temp/checkpoints/best_model.pth")
            
            # Log best model metrics
            writer.add_scalar('Best/Loss', val_loss, epoch)
            writer.add_scalar('Best/Accuracy', val_acc, epoch)
            
            print(f"Saved new best model with validation loss: {val_loss:.6f}")
        
        if epoch % cfg["save_checkpoint_interval"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, f"temp/checkpoints/model_{epoch}.pth")

    print("Training finished.")
    
    # Update hyperparameters with final results
    writer.add_hparams(hparams, {'hparam/best_val_loss': best_val_loss, 'hparam/best_val_acc': best_val_acc})
    
    # Close tensorboard writer
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    print("Training completed successfully!")





