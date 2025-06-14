import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from data_util.surface_sampling_dataloader import get_surface_sampling_dataloader
from pipline.config import config # Import config
from field import MeshSDF
import tools
# from models.trellis.models.sparse_structure_vae import SparseStructureEncoder
import models.trellis.modules.sparse as sp
import models.lzd_models.slat_net as latent_encoder
import models.trellis.models.structured_latent_vae.encoder as trellis_spvae_encoder
import models.trellis.models as trellis_models



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
        encoder_cfg = json.load(open("/mnt/lizd/work/CG/NormalEstimation/iSDF/config/encoder_config.json")).get("model")
        config2 = encoder_cfg.get("SparseLatentEncoder").get("args")
        self.input_dim = in_dim
        self.input_transform = nn.Sequential(
            nn.Linear(self.input_dim, config2.get("in_channels")),
            nn.ReLU(),
            nn.Linear(config2.get("in_channels"), config2.get("in_channels")),
        )
        # TODO: 自制一个encoder（vae的encoder会加噪）
        # self.slat_encoder = latent_encoder.SLatEncoder(**config2)
        
        self.slat_encoder = trellis_spvae_encoder.SLatEncoder(**config2)

    def forward(self, x: sp.SparseTensor):
        feats = x.feats
        feats = self.input_transform(feats)
        assert torch.isnan(feats).any() == False, "Input features contain NaN values"
        x = x.replace(feats)
        return self.slat_encoder(x)


class GridDecoder(nn.Module):
    def __init__(self):
        super(GridDecoder, self).__init__()
        import models.lzd_models.slat_net as decoder_mesh
    
        ss_decoder_cfg = json.load(open("/mnt/lizd/work/CG/NormalEstimation/iSDF/config/decoder_config.json")).get("model").get("SlatVoxelDecoder").get("args")
        
        self.ss_decoder = decoder_mesh.SLatVoxelDecoder(**ss_decoder_cfg)
        self.activation = nn.Sigmoid()

        
    def forward(self, x):
        x = self.ss_decoder(x)
        feats = self.activation(x.feats)
        x = x.replace(feats)
        return x


# --- Placeholder Helper Functions ---
def compute_gt_sdf_grid(vertices, faces, R, device):
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
        mask = tools.create_mask_by_k(query_points.cpu().numpy(),points[i].cpu().numpy(),k=10)
        mask = torch.from_numpy(mask).to(device)
        mask = mask.reshape(grid_shape)
        wnf = wnf.reshape(grid_shape)
        masks[i] = mask
        voxel_grid[i] = wnf
    return voxel_grid,masks # Shape (B, r, r, r)

def check_gradients():
    """检查梯度是否正常"""
    print("\n=== Gradient Check ===")
    
    for name, model in [
                       ('slat_encoder', slat_encoder), 
                       ('grid_decoder', grid_decoder)]:
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

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network for SDF estimation')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (e.g., 0, 1, etc.)')
    args = parser.parse_args()

    # Load configuration
    cfg = config
    
    # Set device based on command-line argument
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    
    print(f"Using device: {device}")

    # Initialize Dataloader
    dataloader = get_surface_sampling_dataloader(
        data_path=cfg["data_path"],
        num_samples=cfg["num_samples"],
        batch_size=cfg["batch_size"],
        save_path=cfg["save_path"],
        use_torch=cfg["use_torch_dataloader"],
        device=device,
        normalize=True
    )

    # # Initialize Models
    # pointnet_encoder = LocalPoolPointnet(
    #     in_channels=cfg["pointnet_input_channels"],
    #     out_channels=cfg["pointnet_output_dim"]
    # ).to(device)


    slat_encoder = SLatEncoder(in_dim=1).to(device)
    grid_decoder = GridDecoder().to(device)

    # Optimizer
    params_to_optimize = (
        list(grid_decoder.parameters()) +
        list(slat_encoder.parameters())
    )
    optimizer = optim.Adam(params_to_optimize, lr=cfg["learning_rate"],eps=1e-4) # 由于是半精度，需要调大eps

    # Loss Function (Mean Squared Error for SDF values)
    loss_fn = nn.L1Loss() # TODO: 使用hou's Loss

    # Training Loop
    
    pointcloud_voxel_resolution = cfg["r"]
    
    for epoch in range(cfg["epochs"]):
        for batch_idx, (points, _, vertices, faces) in enumerate(dataloader):
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
            
            # TODO: important: 坐标转化待验证
            # per_point_feats = pointnet_encoder(points,voxel_coords,res=pointcloud_voxel_resolution)
            # pc_sparse_slat_feats = sp.SparseTensor(per_point_feats,voxel_coords)
            # pc_sparse_slat_feats = torch.zeros(B,1,pointcloud_voxel_resolution,pointcloud_voxel_resolution,pointcloud_voxel_resolution,device=device)

            # 2. Compute GT SDF grid (R x R x R)
            gt_sdf = compute_gt_sdf_grid(vertices, faces, cfg["R"], device) # (B, R, R, R) 
            gt_sdf = gt_sdf.unsqueeze(1)
            gt_sdf[gt_sdf>0.0] = 1
            gt_sdf[gt_sdf<0.0] = 0
            

            # 3. Estimate normals using PCA -> pred_normal
            pred_normals = estimate_normals_pca(points, k=cfg["pca_knn"]) # (B, N_points, 3)

            # 4. Create voxel grid (r x r x r) from predicted normals
            voxel_grid_features,masks = op2wnf(points, pred_normals, cfg["r"], device) # (B, r, r, r)
            indices = torch.nonzero(masks).int()
            feats = voxel_grid_features[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            feats = feats.unsqueeze(1)
            assert not voxel_grid_features.isnan().any()
            
            feats[feats>0.0] = 1.0
            feats[feats<0.0] = 0.0
            
            sp_feats = sp.SparseTensor(feats, indices)
            sp_feats = slat_encoder(sp_feats)
            assert torch.isnan(sp_feats.feats).any() == False, "Input features contain NaN values"
            sp_feats = grid_decoder(sp_feats)
            assert torch.isnan(sp_feats.feats).any() == False, "Input features contain NaN values"
        
            batch_size = points.shape[0]
            indices = sp_feats.coords
            gt_sdf_val = gt_sdf[indices[:,0],:,indices[:,1],indices[:,2],indices[:,3]]
            pred_val = sp_feats.feats
            
            # TODO: 使用mask
            loss = loss_fn(pred_val, gt_sdf_val)
            torch.autograd.set_detect_anomaly(True)
            
            optimizer.zero_grad()
            # 反向传播时检测是否有异常值，定位code
            with torch.autograd.detect_anomaly():
                loss.backward()
            with torch.autograd.detect_anomaly():
                optimizer.step()

            if batch_idx % cfg["log_interval"] == 0:
                acc = (pred_val>0.5).eq(gt_sdf_val>0.5).float().mean()
                print(f"Epoch {epoch+1}/{cfg['epochs']}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.6f}, Acc: {acc.item():.6f}")

        print(f"Epoch {epoch+1}/{cfg['epochs']} completed. Last batch loss: {loss.item():.6f}")

    print("Training finished.")





