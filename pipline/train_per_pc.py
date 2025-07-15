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
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import cc3d


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
        encoder_cfg = json.load(open("config/encoder_config.json")).get("model").get("SparseLatentEncoder").get("args")
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
    
        ss_decoder_cfg = json.load(open("config/decoder_config.json")).get("model").get("SlatVoxelDecoder").get("args")
        
        self.ss_decoder = lat_net.SLatVoxelDecoder(**ss_decoder_cfg)
        self.activation = nn.Tanh()

        
    def forward(self, x):
        x = self.ss_decoder(x)
        feats = self.activation(x.feats)
        x = x.replace(feats)
        return x


class VoxelGridVAEBuilder(nn.Module):
    def __init__(self):
        super(VoxelGridVAEBuilder, self).__init__()
        config = json.load(open("config/voxelGridVAE.json")).get("model")    
        input_channels = config.get("input_channels")
        output_channels = config.get("output_channels")
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

# 自定义loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
    def forward(self, pred, target):
        loss = 1 - torch.abs(torch.mean(pred*target))
        return loss * 1000

class MyL1Loss(nn.Module):
    def __init__(self):
        super(MyL1Loss, self).__init__()
        self.loss_fn = nn.L1Loss()
        
    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        return loss*1000

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

# def evaluate_model(model, dataloader, device, cfg, loss_fn, epoch=None, save_results=False):
#     """
#     Evaluates the model on the validation/test set using **exactly** the same
#     data-preparation pipeline as the training loop so that no distribution
#     shift is introduced between training and evaluation.
#     """
#     model.eval()

#     total_loss = 0.0
#     total_acc = 0.0
#     total_acc_wnf = 0.0
#     total_feats_acc = 0.0
#     num_batches = 0

#     with torch.no_grad():
#         for batch_idx, (points, gt_normals, vertices, faces,file_name) in enumerate(dataloader):
#             # Ensure data is on the correct device
#             points = points.to(device)         # (B, N_points, 3)
#             B = points.size(0)

#             # === Use EXACTLY the same data processing pipeline as training ===
            
#             # 1. Estimate normals with PCA (same as training)
#             pred_normals = estimate_normals_pca(points, k=cfg["pca_knn"]) # (B, N_points, 3)

#             # 2. Create uniform grid and initialize tensors (same as training)
#             i_resolu = cfg["r"]
#             o_resolu = cfg["R"]
#             voxel_center,grid_shape = tools.create_uniform_grid(i_resolu,bbox=np.array([[-1,1],[-1,1],[-1,1]]))
#             feat_shape = [B] + list(grid_shape)
#             gt_wnf = torch.zeros(feat_shape,device=device)
#             pred_wnf = torch.zeros(feat_shape,device=device)
#             gt_sdf = torch.zeros(feat_shape,device=device)
#             gt_udf = torch.zeros(feat_shape,device=device)
#             pred_udf = torch.zeros(feat_shape,device=device)
#             masks = torch.zeros(feat_shape,device=device)
            
#             # 3. Process each batch item (same as training)
#             import cal_wnf
#             sdf_field = SDFField(i_resolu)
#             for i in range(B):
#                 mask = tools.create_mask_by_k(voxel_center,points[i].cpu().numpy(),k=60)    
#                 query_points_i = voxel_center[mask]
#                 mask_cuda = torch.from_numpy(mask).to(device).reshape(grid_shape)
#                 masks[i] = mask_cuda
#                 mask = mask.reshape(grid_shape)
#                 query_points_i = torch.from_numpy(query_points_i).to(device)
#                 pred_normals_i = pred_normals[i]
                
#                 # 计算wnf (same as training)
#                 wnf_i = cal_wnf.compute_winding_number_torch_api(points[i],pred_normals[i],query_points_i,epsilon=1e-8,batch_size=10000)
#                 gt_wnf_i = cal_wnf.compute_winding_number_torch_api(points[i],gt_normals[i],query_points_i,epsilon=1e-8,batch_size=10000)
#                 pred_wnf_grad_i = cal_wnf.compute_winding_number_torch_api(points[i],pred_normals[i],query_points_i,epsilon=1e-8,batch_size=10000)
                
#                 pred_wnf[i][mask_cuda] = wnf_i
#                 gt_wnf[i][mask_cuda] = gt_wnf_i
#                 pred_wnf_grad[i][mask_cuda] = pred_wnf_grad_i
#                 # 计算sdf (same as training)
#                 mesh_sdf = MeshSDF(vertices[i].cpu().numpy(), faces[i].cpu().numpy())
#                 sdf_values = mesh_sdf.query(query_points_i.cpu().numpy())
#                 gt_sdf[i][mask_cuda] = torch.from_numpy(sdf_values).to(device,dtype=torch.float32)
                
#                 # 计算udf (same as training)
#                 gt_udf[i][mask_cuda] = torch.abs(gt_sdf[i][mask_cuda])
#                 pred_udf[i][mask_cuda] = torch.abs(sdf_field.compute_sdf(points[i],pred_normals[i],query_points_i))
                
#             # 4. Gather sparse indices and values (same as training)       
#             indices = torch.nonzero(masks).int()
#             gt_sdf_val = gt_sdf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
#             gt_udf_val = gt_udf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
#             pred_udf_val = pred_udf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
#             wnf_val = pred_wnf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
#             gt_wnf_val = gt_wnf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
#             pred_wnf_grad_val = pred_wnf_grad[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
#             wnf_val = torch.tanh(wnf_val)
#             gt_wnf_val = torch.tanh(gt_wnf_val)
#             pred_wnf_grad_val = torch.tanh(pred_wnf_grad_val)
            
#             # 5. Create features (same as training)
#             feats1 = wnf_val
#             feats1 = torch.tanh(feats1)
#             feats_acc = (feats1>0.0).squeeze().eq(gt_wnf_val>0.0).float().mean()
#             feats12 = torch.cat([feats1.unsqueeze(1),pred_udf_val.unsqueeze(1)],dim=1)
            
#             # 6. Forward pass through model (same as training)
#             # sp_feats = sp.SparseTensor(feats12, indices)
#             sp_feats = sp.SparseTensor(pred_wnf_grad_val.squeeze().unsqueeze(1),indices)
#             sp_feats = model(sp_feats)
#             pred_val = sp_feats.feats.squeeze()
            
#             # 7. Calculate metrics (same as training)
#             loss = loss_fn(pred_val, gt_wnf_val)
#             acc = (pred_val>0.0).eq(gt_wnf_val>0.0).float().mean()
#             loss_gt_wnf = loss_fn(gt_wnf_val,gt_sdf_val)
#             acc_wnf = (gt_sdf_val>0.0).eq(gt_wnf_val>0.0).float().mean()

#             total_loss += loss.item()
#             total_acc += acc.item()
#             total_acc_wnf += acc_wnf.item()
#             total_feats_acc += feats_acc.item()
#             num_batches += 1
            
#             # 8. Save validation results if requested
#             if save_results and epoch is not None:
#                 val_save_dir = f"val_results/epoch_{epoch}"
#                 os.makedirs(val_save_dir, exist_ok=True)
                
#                 # For each item in the batch
#                 for i in range(B):
#                     batch_save_dir = f"{val_save_dir}/batch_{batch_idx}_item_{i}"
#                     os.makedirs(batch_save_dir, exist_ok=True)
                    
#                     # Create full prediction SDF grid
#                     pred_sdf = torch.zeros_like(gt_sdf)
#                     pred_sdf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]] = pred_val
                    
#                     # Save meshes
#                     try:
#                         # Extract and save GT SDF surface
#                         tools.extract_surface_from_scalar_field(
#                             gt_sdf[i].squeeze().cpu().numpy(), 
#                             level=0, 
#                             resolution=cfg["r"], 
#                             save_path=f"{batch_save_dir}/gt_sdf.ply",
#                             mask=masks[i].cpu().numpy()
#                         )
                        
#                         # Extract and save predicted SDF surface
#                         tools.extract_surface_from_scalar_field(
#                             pred_sdf[i].squeeze().cpu().detach().numpy(), 
#                             level=0, 
#                             resolution=cfg["r"], 
#                             save_path=f"{batch_save_dir}/pred_sdf.ply",
#                             mask=masks[i].cpu().numpy()
#                         )
                        
#                         # Extract and save GT WNF surface
#                         tools.extract_surface_from_scalar_field(
#                             gt_wnf[i].squeeze().cpu().numpy(), 
#                             level=0, 
#                             resolution=cfg["r"], 
#                             save_path=f"{batch_save_dir}/gt_wnf.ply",
#                             mask=masks[i].cpu().numpy()
#                         )
                        
#                         # Save original GT mesh
#                         import open3d as o3d
#                         o3d.io.write_triangle_mesh(
#                             f"{batch_save_dir}/gt_mesh.ply",
#                             o3d.geometry.TriangleMesh(
#                                 o3d.utility.Vector3dVector(vertices[i].cpu().numpy()),
#                                 o3d.utility.Vector3iVector(faces[i].cpu().numpy())
#                             )
#                         )
                        
#                         # Save point cloud with normals
#                         pcd = o3d.geometry.PointCloud()
#                         pcd.points = o3d.utility.Vector3dVector(points[i].cpu().numpy())
#                         pcd.normals = o3d.utility.Vector3dVector(pred_normals[i].cpu().numpy())
#                         o3d.io.write_point_cloud(f"{batch_save_dir}/input_points_with_normals.ply", pcd)
                        
#                     except Exception as e:
#                         print(f"Warning: Could not save mesh for validation batch {batch_idx}, item {i}: {e}")
                    
#                     # Save scalar fields as numpy arrays
#                     try:
#                         np.save(f"{batch_save_dir}/gt_sdf.npy", gt_sdf[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/pred_sdf.npy", pred_sdf[i].cpu().detach().numpy())
#                         np.save(f"{batch_save_dir}/gt_wnf.npy", gt_wnf[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/pred_wnf.npy", pred_wnf[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/gt_udf.npy", gt_udf[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/pred_udf.npy", pred_udf[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/mask.npy", masks[i].cpu().numpy())
                        
#                         # Save input data
#                         np.save(f"{batch_save_dir}/input_points.npy", points[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/pred_normals.npy", pred_normals[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/gt_normals.npy", gt_normals[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/vertices.npy", vertices[i].cpu().numpy())
#                         np.save(f"{batch_save_dir}/faces.npy", faces[i].cpu().numpy())
                        
#                     except Exception as e:
#                         print(f"Warning: Could not save numpy arrays for validation batch {batch_idx}, item {i}: {e}")
                    
#                     # Save metrics for this sample
#                     try:
#                         metrics = {
#                             'loss': loss.item(),
#                             'accuracy': acc.item(),
#                             'accuracy_wnf': acc_wnf.item(),
#                             'feats_accuracy': feats_acc.item(),
#                             'loss_gt_wnf': loss_gt_wnf.item(),
#                             'num_valid_voxels': masks[i].sum().item(),
#                             'total_voxels': masks[i].numel()
#                         }
                        
#                         with open(f"{batch_save_dir}/metrics.json", 'w') as f:
#                             json.dump(metrics, f, indent=2)
                            
#                     except Exception as e:
#                         print(f"Warning: Could not save metrics for validation batch {batch_idx}, item {i}: {e}")
                
#                 # Only save first few batches to avoid too much storage
#                 if batch_idx >= 2:  # Save first 3 batches
#                     break

#     # 9. Aggregate statistics
#     avg_loss = total_loss / max(num_batches, 1)
#     avg_acc = total_acc / max(num_batches, 1)
#     avg_acc_wnf = total_acc_wnf / max(num_batches, 1)
#     avg_feats_acc = total_feats_acc / max(num_batches, 1)

#     return avg_loss, avg_acc, avg_acc_wnf, avg_feats_acc

class TrainingVisualizer:
    """统一管理训练过程中的可视化和保存"""
    
    def __init__(self, cfg, log_dir, dataset_name=None):
        self.cfg = cfg
        self.dataset_name = dataset_name or "unknown"
        
        # 创建主输出目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"experiment_results/{self.dataset_name}_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.train_dir = self.output_dir / "training"
        self.val_dir = self.output_dir / "validation" 
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.analysis_dir = self.output_dir / "analysis"
        self.config_dir = self.output_dir / "config"
        
        for dir_path in [self.train_dir, self.val_dir, self.checkpoint_dir, self.analysis_dir, self.config_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 保存配置文件
        with open(self.config_dir / "config.json", 'w') as f:
            json.dump(cfg, f, indent=2)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 训练历史记录
        self.train_history = {
            'epoch': [], 'batch': [], 'loss': [], 'accuracy': [], 
            'feats_accuracy': [], 'grad_norm_encoder': [], 'grad_norm_decoder': []
        }
        self.val_history = {
            'epoch': [], 'loss': [], 'accuracy': [], 'accuracy_wnf': [], 'feats_accuracy': []
        }
        
        print(f"📁 Experiment results will be saved to: {self.output_dir}")
        print(f"📊 TensorBoard logs: {log_dir}")
        
    def save_training_batch_results(self, epoch, batch_idx, loss_val, acc_val, feats_acc_val, 
                                  grad_norm_encoder, grad_norm_decoder, global_step,
                                  gt_sdf, pred_sdf, gt_wnf, pred_wnf_grad, masks, 
                                  vertices, faces, points, pred_normals, gt_normals, file_names=None):
        """保存训练批次结果"""
        
        # 记录历史
        self.train_history['epoch'].append(epoch)
        self.train_history['batch'].append(batch_idx)
        self.train_history['loss'].append(loss_val)
        self.train_history['accuracy'].append(acc_val)
        self.train_history['feats_accuracy'].append(feats_acc_val)
        self.train_history['grad_norm_encoder'].append(grad_norm_encoder)
        self.train_history['grad_norm_decoder'].append(grad_norm_decoder)
        
        # TensorBoard日志
        self.writer.add_scalar('Train/Loss', loss_val, global_step)
        self.writer.add_scalar('Train/Accuracy', acc_val, global_step)
        self.writer.add_scalar('Train/Feats_Accuracy', feats_acc_val, global_step)
        self.writer.add_scalar('Train/GradNorm_Encoder', grad_norm_encoder, global_step)
        self.writer.add_scalar('Train/GradNorm_Decoder', grad_norm_decoder, global_step)
        
        # 每隔一定间隔保存详细结果
        if batch_idx % self.cfg.get("save_detailed_interval", 50) == 0:
            self._save_detailed_batch_results(
                epoch, batch_idx, "train", gt_sdf, pred_sdf, gt_wnf, pred_wnf_grad, 
                masks, vertices, faces, points, pred_normals, gt_normals, global_step, file_names
            )
    
    def save_validation_epoch_results(self, epoch, val_loss, val_acc, val_acc_wnf, val_feats_acc):
        """保存验证epoch结果"""
        
        # 记录历史
        self.val_history['epoch'].append(epoch)
        self.val_history['loss'].append(val_loss)
        self.val_history['accuracy'].append(val_acc)
        self.val_history['accuracy_wnf'].append(val_acc_wnf)
        self.val_history['feats_accuracy'].append(val_feats_acc)
        
        # TensorBoard日志
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
        self.writer.add_scalar('Val/Accuracy_WNF', val_acc_wnf, epoch)
        self.writer.add_scalar('Val/Feats_Accuracy', val_feats_acc, epoch)
        
        # 保存历史记录
        self._save_training_history()
        
        # 生成可视化图表
        self._generate_training_plots()
        
    def _save_detailed_batch_results(self, epoch, batch_idx, phase, gt_sdf, pred_sdf, gt_wnf, 
                                   pred_wnf_grad, masks, vertices, faces, points, pred_normals, 
                                   gt_normals, global_step, file_names=None):
        """保存详细的批次结果"""
        
        # 选择保存目录
        save_dir = self.train_dir if phase == "train" else self.val_dir
        
        # 使用文件名创建目录，如果没有文件名则使用batch_idx
        if file_names is not None and len(file_names) > 0:
            # 清理文件名，移除扩展名和特殊字符
            clean_filename = Path(file_names[0]).stem
            clean_filename = "".join(c for c in clean_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            dir_name = clean_filename if clean_filename else f"batch_{batch_idx:03d}"
        else:
            dir_name = f"batch_{batch_idx:03d}"
            
        batch_dir = save_dir / f"epoch_{epoch:03d}" / dir_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存多个样本而不是只保存第一个
        batch_size = gt_sdf.shape[0]
        num_samples_to_save = min(batch_size, 2)  # 最多保存前2个样本
        
        for sample_idx in range(num_samples_to_save):
            sample_dir = batch_dir / f"sample_{sample_idx}"
            sample_dir.mkdir(exist_ok=True)
            
            # 获取当前样本的文件名
            current_filename = file_names[sample_idx] if file_names and sample_idx < len(file_names) else f"sample_{sample_idx}"
            
            try:
                # 保存网格文件
                mesh_dir = sample_dir / "meshes"
                mesh_dir.mkdir(exist_ok=True)
                
                # GT SDF表面
                tools.extract_surface_from_scalar_field(
                    gt_sdf[sample_idx].squeeze().cpu().numpy(),
                    level=0, resolution=self.cfg["r"],
                    save_path=str(mesh_dir / "gt_sdf_surface.ply"),
                    mask=masks[sample_idx].cpu().numpy()
                )
                
                # 预测SDF表面
                tools.extract_surface_from_scalar_field(
                    pred_sdf[sample_idx].squeeze().cpu().detach().numpy(),
                    level=0, resolution=self.cfg["r"],
                    save_path=str(mesh_dir / "pred_sdf_surface.ply"),
                    mask=masks[sample_idx].cpu().numpy()
                )
                
                # GT WNF表面
                tools.extract_surface_from_scalar_field(
                    gt_wnf[sample_idx].squeeze().cpu().numpy(),
                    level=0, resolution=self.cfg["r"],
                    save_path=str(mesh_dir / "gt_wnf_surface.ply"),
                    mask=masks[sample_idx].cpu().numpy()
                )
                
                # 原始网格
                import open3d as o3d
                o3d.io.write_triangle_mesh(
                    str(mesh_dir / "original_mesh.ply"),
                    o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(vertices[sample_idx].cpu().numpy()),
                        o3d.utility.Vector3iVector(faces[sample_idx].cpu().numpy())
                    )
                )
                
                # 带法向量的点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[sample_idx].cpu().numpy())
                pcd.normals = o3d.utility.Vector3dVector(pred_normals[sample_idx].cpu().numpy())
                o3d.io.write_point_cloud(str(mesh_dir / "input_points_normals.ply"), pcd)
                
            except Exception as e:
                print(f"Warning: Could not save meshes for {phase} epoch {epoch}, {dir_name}, sample {sample_idx}: {e}")
            
            try:
                # 保存标量场
                fields_dir = sample_dir / "fields"
                fields_dir.mkdir(exist_ok=True)
                
                np.save(fields_dir / "gt_sdf.npy", gt_sdf[sample_idx].cpu().numpy())
                np.save(fields_dir / "pred_sdf.npy", pred_sdf[sample_idx].cpu().detach().numpy())
                np.save(fields_dir / "gt_wnf.npy", gt_wnf[sample_idx].cpu().numpy())
                np.save(fields_dir / "pred_wnf_grad.npy", pred_wnf_grad[sample_idx].cpu().numpy())
                np.save(fields_dir / "mask.npy", masks[sample_idx].cpu().numpy())
                
            except Exception as e:
                print(f"Warning: Could not save fields for {phase} epoch {epoch}, {dir_name}, sample {sample_idx}: {e}")
            
            try:
                # 保存标量场的切片图像
                self._save_field_slice_images(
                    gt_sdf[sample_idx], pred_sdf[sample_idx], gt_wnf[sample_idx], 
                    pred_wnf_grad[sample_idx], masks[sample_idx], sample_dir, current_filename
                )
                
            except Exception as e:
                print(f"Warning: Could not save images for {phase} epoch {epoch}, {dir_name}, sample {sample_idx}: {e}")
        
        # TensorBoard可视化 - 添加切片图像（只为第一个样本）
        if phase == "train" and batch_idx % self.cfg.get("tensorboard_vis_interval", 100) == 0:
            self._add_field_slices_to_tensorboard(
                gt_sdf[0], pred_sdf[0], gt_wnf[0], 
                pred_wnf_grad[0], global_step
            )
    
    def _add_field_slices_to_tensorboard(self, gt_sdf, pred_sdf, gt_wnf, pred_wnf_grad, global_step):
        """添加标量场切片到TensorBoard"""
        mid_slice = self.cfg["r"] // 2
        
        # 归一化用于可视化
        def normalize_for_vis(tensor):
            tensor = tensor.detach().cpu().numpy()
            return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        
        self.writer.add_image('Fields/GT_SDF_Slice', 
                            normalize_for_vis(gt_sdf[mid_slice:mid_slice+1, :, :]), 
                            global_step, dataformats='CHW')
        self.writer.add_image('Fields/Pred_SDF_Slice', 
                            normalize_for_vis(pred_sdf[mid_slice:mid_slice+1, :, :]), 
                            global_step, dataformats='CHW')
        self.writer.add_image('Fields/GT_WNF_Slice', 
                            normalize_for_vis(gt_wnf[mid_slice:mid_slice+1, :, :]), 
                            global_step, dataformats='CHW')
        self.writer.add_image('Fields/Pred_WNF_Grad_Slice', 
                            normalize_for_vis(pred_wnf_grad[mid_slice:mid_slice+1, :, :]), 
                            global_step, dataformats='CHW')
    
    def _save_field_slice_images(self, gt_sdf, pred_sdf, gt_wnf, pred_wnf_grad, mask, save_dir, filename):
        """保存标量场的切片图像"""
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        
        # 创建图像保存目录
        images_dir = save_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # 获取数据
        gt_sdf_data = gt_sdf.detach().cpu().numpy()
        pred_sdf_data = pred_sdf.detach().cpu().numpy()
        gt_wnf_data = gt_wnf.detach().cpu().numpy()
        pred_wnf_grad_data = pred_wnf_grad.detach().cpu().numpy()
        mask_data = mask.cpu().numpy()
        
        resolution = gt_sdf_data.shape[0]
        mid_slice = resolution // 2
        
        # 定义要保存的切片
        slices_to_save = [
            resolution // 4,
            mid_slice,
            3 * resolution // 4
        ]
        
        for axis in range(3):  # X, Y, Z轴
            axis_names = ['X', 'Y', 'Z']
            
            for slice_idx in slices_to_save:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'{filename} - {axis_names[axis]}-axis Slice {slice_idx}', fontsize=16)
                
                # 根据轴选择切片
                if axis == 0:  # X轴切片
                    gt_sdf_slice = gt_sdf_data[slice_idx, :, :]
                    pred_sdf_slice = pred_sdf_data[slice_idx, :, :]
                    gt_wnf_slice = gt_wnf_data[slice_idx, :, :]
                    pred_wnf_grad_slice = pred_wnf_grad_data[slice_idx, :, :]
                    mask_slice = mask_data[slice_idx, :, :]
                elif axis == 1:  # Y轴切片
                    gt_sdf_slice = gt_sdf_data[:, slice_idx, :]
                    pred_sdf_slice = pred_sdf_data[:, slice_idx, :]
                    gt_wnf_slice = gt_wnf_data[:, slice_idx, :]
                    pred_wnf_grad_slice = pred_wnf_grad_data[:, slice_idx, :]
                    mask_slice = mask_data[:, slice_idx, :]
                else:  # Z轴切片
                    gt_sdf_slice = gt_sdf_data[:, :, slice_idx]
                    pred_sdf_slice = pred_sdf_data[:, :, slice_idx]
                    gt_wnf_slice = gt_wnf_data[:, :, slice_idx]
                    pred_wnf_grad_slice = pred_wnf_grad_data[:, :, slice_idx]
                    mask_slice = mask_data[:, :, slice_idx]
                
                # 应用mask
                gt_sdf_slice = np.where(mask_slice, gt_sdf_slice, np.nan)
                pred_sdf_slice = np.where(mask_slice, pred_sdf_slice, np.nan)
                gt_wnf_slice = np.where(mask_slice, gt_wnf_slice, np.nan)
                pred_wnf_grad_slice = np.where(mask_slice, pred_wnf_grad_slice, np.nan)
                
                # GT SDF
                im1 = axes[0, 0].imshow(gt_sdf_slice, cmap='RdBu_r', origin='lower')
                axes[0, 0].set_title('GT SDF')
                axes[0, 0].contour(gt_sdf_slice, levels=[0], colors='black', linewidths=2)
                plt.colorbar(im1, ax=axes[0, 0], shrink=0.6)
                
                # Predicted SDF
                im2 = axes[0, 1].imshow(pred_sdf_slice, cmap='RdBu_r', origin='lower')
                axes[0, 1].set_title('Predicted SDF')
                axes[0, 1].contour(pred_sdf_slice, levels=[0], colors='black', linewidths=2)
                plt.colorbar(im2, ax=axes[0, 1], shrink=0.6)
                
                # GT WNF
                im3 = axes[1, 0].imshow(gt_wnf_slice, cmap='viridis', origin='lower')
                axes[1, 0].set_title('GT WNF')
                axes[1, 0].contour(gt_wnf_slice, levels=[0], colors='red', linewidths=2)
                plt.colorbar(im3, ax=axes[1, 0], shrink=0.6)
                
                # Predicted WNF Gradient
                im4 = axes[1, 1].imshow(pred_wnf_grad_slice, cmap='viridis', origin='lower')
                axes[1, 1].set_title('Predicted WNF Gradient')
                axes[1, 1].contour(pred_wnf_grad_slice, levels=[0], colors='red', linewidths=2)
                plt.colorbar(im4, ax=axes[1, 1], shrink=0.6)
                
                # 保存图像
                plt.tight_layout()
                save_path = images_dir / f'slice_{axis_names[axis].lower()}_{slice_idx:03d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        # 保存3D概览图（显示mask的范围）
        try:
            fig = plt.figure(figsize=(15, 5))
            
            # X-Y平面 (中间Z切片)
            ax1 = fig.add_subplot(131)
            mask_xy = mask_data[:, :, mid_slice]
            ax1.imshow(mask_xy, cmap='gray', origin='lower')
            ax1.set_title('Mask - XY plane (mid Z)')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            
            # X-Z平面 (中间Y切片)
            ax2 = fig.add_subplot(132)
            mask_xz = mask_data[:, mid_slice, :]
            ax2.imshow(mask_xz, cmap='gray', origin='lower')
            ax2.set_title('Mask - XZ plane (mid Y)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Z')
            
            # Y-Z平面 (中间X切片)
            ax3 = fig.add_subplot(133)
            mask_yz = mask_data[mid_slice, :, :]
            ax3.imshow(mask_yz, cmap='gray', origin='lower')
            ax3.set_title('Mask - YZ plane (mid X)')
            ax3.set_xlabel('Y')
            ax3.set_ylabel('Z')
            
            plt.tight_layout()
            plt.savefig(images_dir / 'mask_overview.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not save mask overview: {e}")
    
    def _save_training_history(self):
        """保存训练历史到CSV文件"""
        train_df = pd.DataFrame(self.train_history)
        val_df = pd.DataFrame(self.val_history)
        
        train_df.to_csv(self.analysis_dir / "train_history.csv", index=False)
        val_df.to_csv(self.analysis_dir / "val_history.csv", index=False)
    
    def _generate_training_plots(self):
        """生成训练过程可视化图表"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            # Fallback to default style if seaborn is not available
            plt.style.use('default')
        
        # 训练损失和准确率
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Progress - {self.dataset_name}', fontsize=16)
        
        # 训练损失
        if len(self.train_history['loss']) > 0:
            axes[0, 0].plot(self.train_history['loss'], 'b-', alpha=0.7, label='Training Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            axes[0, 0].legend()
        
        # 训练准确率
        if len(self.train_history['accuracy']) > 0:
            axes[0, 1].plot(self.train_history['accuracy'], 'g-', alpha=0.7, label='Training Accuracy')
            axes[0, 1].plot(self.train_history['feats_accuracy'], 'r-', alpha=0.7, label='Features Accuracy')
            axes[0, 1].set_title('Training Accuracy')
            axes[0, 1].set_xlabel('Batch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True)
            axes[0, 1].legend()
        
        # 验证指标
        if len(self.val_history['loss']) > 0:
            axes[1, 0].plot(self.val_history['epoch'], self.val_history['loss'], 'bo-', label='Val Loss')
            ax_twin = axes[1, 0].twinx()
            ax_twin.plot(self.val_history['epoch'], self.val_history['accuracy'], 'ro-', label='Val Accuracy')
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss', color='b')
            ax_twin.set_ylabel('Accuracy', color='r')
            axes[1, 0].grid(True)
            axes[1, 0].legend(loc='upper left')
            ax_twin.legend(loc='upper right')
        
        # 梯度范数
        if len(self.train_history['grad_norm_encoder']) > 0:
            axes[1, 1].plot(self.train_history['grad_norm_encoder'], 'c-', alpha=0.7, label='Encoder Grad Norm')
            axes[1, 1].plot(self.train_history['grad_norm_decoder'], 'm-', alpha=0.7, label='Decoder Grad Norm')
            axes[1, 1].set_title('Gradient Norms')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Grad Norm')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 验证详细分析
        if len(self.val_history['epoch']) > 1:
            self._generate_validation_analysis()
    
    def _generate_validation_analysis(self):
        """生成详细的验证分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Validation Analysis', fontsize=16)
        
        epochs = self.val_history['epoch']
        
        # 验证损失趋势
        axes[0, 0].plot(epochs, self.val_history['loss'], 'b-o', markersize=4)
        axes[0, 0].set_title('Validation Loss Trend')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 多个准确率指标对比
        axes[0, 1].plot(epochs, self.val_history['accuracy'], 'g-o', label='SDF Accuracy', markersize=4)
        axes[0, 1].plot(epochs, self.val_history['accuracy_wnf'], 'r-s', label='WNF Accuracy', markersize=4)
        axes[0, 1].plot(epochs, self.val_history['feats_accuracy'], 'b-^', label='Features Accuracy', markersize=4)
        axes[0, 1].set_title('Validation Accuracies Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 最佳性能标记
        best_loss_idx = np.argmin(self.val_history['loss'])
        best_acc_idx = np.argmax(self.val_history['accuracy'])
        
        axes[1, 0].bar(['Best Loss Epoch', 'Best Acc Epoch'], 
                      [epochs[best_loss_idx], epochs[best_acc_idx]],
                      color=['lightcoral', 'lightgreen'])
        axes[1, 0].set_title('Best Performance Epochs')
        axes[1, 0].set_ylabel('Epoch')
        
        # 性能统计
        stats_text = f"""
        Best Validation Loss: {min(self.val_history['loss']):.6f} (Epoch {epochs[best_loss_idx]})
        Best Validation Accuracy: {max(self.val_history['accuracy']):.6f} (Epoch {epochs[best_acc_idx]})
        Final Loss: {self.val_history['loss'][-1]:.6f}
        Final Accuracy: {self.val_history['accuracy'][-1]:.6f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "validation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch, model, optimizer, val_loss, val_acc, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.cfg
        }
        
        # 常规检查点
        if epoch % self.cfg.get("save_checkpoint_interval", 10) == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"model_epoch_{epoch:03d}.pth")
        
        # 最佳模型
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            print(f"💾 Saved best model with validation loss: {val_loss:.6f}")
        
        # 最新模型（总是保存）
        torch.save(checkpoint, self.checkpoint_dir / "latest_model.pth")
    
    def generate_final_report(self):
        """生成最终训练报告"""
        report = {
            "experiment_info": {
                "dataset": self.dataset_name,
                "config": self.cfg,
                "total_training_batches": len(self.train_history['loss']),
                "total_epochs": max(self.val_history['epoch']) if self.val_history['epoch'] else 0
            },
            "best_performance": {
                "best_val_loss": min(self.val_history['loss']) if self.val_history['loss'] else float('inf'),
                "best_val_accuracy": max(self.val_history['accuracy']) if self.val_history['accuracy'] else 0.0,
                "best_loss_epoch": self.val_history['epoch'][np.argmin(self.val_history['loss'])] if self.val_history['loss'] else 0,
                "best_acc_epoch": self.val_history['epoch'][np.argmax(self.val_history['accuracy'])] if self.val_history['accuracy'] else 0
            },
            "final_performance": {
                "final_val_loss": self.val_history['loss'][-1] if self.val_history['loss'] else float('inf'),
                "final_val_accuracy": self.val_history['accuracy'][-1] if self.val_history['accuracy'] else 0.0
            }
        }
        
        with open(self.analysis_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        print(f"📄 Training report saved to: {self.analysis_dir}")
        return report
    
    def _generate_markdown_report(self, report):
        """生成Markdown格式的训练报告"""
        markdown_content = f"""
# Training Report - {self.dataset_name}

## Experiment Configuration
- **Dataset**: {report['experiment_info']['dataset']}
- **Total Epochs**: {report['experiment_info']['total_epochs']}
- **Total Training Batches**: {report['experiment_info']['total_training_batches']}
- **Resolution**: {self.cfg.get('r', 'N/A')}
- **Batch Size**: {self.cfg.get('batch_size', 'N/A')}
- **Learning Rate**: {self.cfg.get('learning_rate', 'N/A')}

## Best Performance
- **Best Validation Loss**: {report['best_performance']['best_val_loss']:.6f} (Epoch {report['best_performance']['best_loss_epoch']})
- **Best Validation Accuracy**: {report['best_performance']['best_val_accuracy']:.6f} (Epoch {report['best_performance']['best_acc_epoch']})

## Final Performance  
- **Final Validation Loss**: {report['final_performance']['final_val_loss']:.6f}
- **Final Validation Accuracy**: {report['final_performance']['final_val_accuracy']:.6f}

## Files Structure
```
{self.output_dir}/
├── training/           # Training batch results
├── validation/         # Validation results  
├── checkpoints/        # Model checkpoints
├── analysis/          # Training plots and statistics
└── config/            # Experiment configuration
```

## Key Files
- `analysis/training_progress.png` - Training curves
- `analysis/validation_analysis.png` - Validation analysis
- `checkpoints/best_model.pth` - Best performing model
- `analysis/train_history.csv` - Training metrics history
- `analysis/val_history.csv` - Validation metrics history

Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        with open(self.analysis_dir / "README.md", 'w') as f:
            f.write(markdown_content)
    
    def close(self):
        """关闭可视化器"""
        self.writer.close()
        final_report = self.generate_final_report()
        print(f"✅ Training visualization complete. Results saved to: {self.output_dir}")
        return final_report

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
    vae = VoxelGridVAEBuilder().to(device)    


    # Optimizer
    params_to_optimize = (
        list(vae.parameters())
    )
    optimizer = optim.Adam(params_to_optimize, lr=cfg["learning_rate"], eps=1e-4)

    # Loss Function
    if cfg["loss_fn"] == "CustomLoss":
        loss_fn = CustomLoss()
    elif cfg["loss_fn"] == "MyL1Loss":
        loss_fn = MyL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {cfg['loss_fn']}")
    
    # Initialize Training Visualizer
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/train_{args.dataset_name}_{timestamp}" if args.dataset_name else f"runs/train_{timestamp}"
    
    # Log hyperparameters to config
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
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(cfg, log_dir, args.dataset_name)
    
    # Training Loop
    pointcloud_voxel_resolution = cfg["r"]
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # Add model architecture to tensorboard (will be done on first forward pass)
    model_logged = False
    
    for epoch in range(cfg["epochs"]):
        # Training phase
        vae.train()
        
        for batch_idx, (points, gt_normals, vertices, faces, file_name) in enumerate(train_dataloader):
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
            vae.train()   

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
            pred_wnf_grad = torch.zeros(feat_shape,device=device)
            masks = torch.zeros(feat_shape,device=device)
            
            import cal_wnf
            sdf_field = SDFField(i_resolu)
            for i in range(B):
                mask = tools.create_mask_by_k(voxel_center,points[i].cpu().numpy(),k=cfg["k_for_mask"])    
                t  = cc3d.connected_components(mask.reshape(grid_shape))
                if t.max() > 1:
                    print("Warning: connected components num of mask = {}".format(t.max()))
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
                wnf_grad = tools.compute_gradient(pred_wnf[i],mask_cuda)
                pred_wnf_grad[i][mask_cuda] = wnf_grad[mask_cuda]
                
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
            pred_wnf_grad_val = pred_wnf_grad[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            gt_wnf_val = gt_wnf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
            wnf_val = torch.tanh(wnf_val)
            gt_wnf_val = torch.tanh(gt_wnf_val)
            pred_wnf_grad_val = torch.tanh(pred_wnf_grad_val)
            
            feats1 = wnf_val
            feats1 = torch.tanh(feats1)
            feats_acc = (feats1>0.0).squeeze().eq(gt_wnf_val>0.0).float().mean()
            # feats12 = torch.cat([feats1.unsqueeze(1),pred_udf_val.unsqueeze(1)],dim=1)
            feats12 = torch.cat([feats1.unsqueeze(1),pred_wnf_grad_val.unsqueeze(1)],dim=1)
            # sp_feats = sp.SparseTensor(feats12, indices)
            pred_wnf_grad_val = pred_wnf_grad_val.squeeze().unsqueeze(1)
            sp_feats = sp.SparseTensor(pred_wnf_grad_val,indices)
            sp_feats = vae(sp_feats)
            pred_val = sp_feats.feats.squeeze()
            
            # Log model graph to tensorboard (only once)
            if not model_logged and batch_idx == 0 and epoch == 0:
                try:
                    # Create a sample input for graph logging
                    sample_input = sp.SparseTensor(feats12[:100], indices[:100])  # Use first 100 points
                    visualizer.writer.add_graph(vae, sample_input)
                    model_logged = True
                    print("Model graph added to TensorBoard")
                except Exception as e:
                    print(f"Could not add model graph to TensorBoard: {e}")
                    model_logged = True  # Don't try again
            
            loss = loss_fn(pred_val, gt_wnf_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_grad_norm_encoder = get_total_grad_norm(vae.model.ss_encoder)
            total_grad_norm_decoder = get_total_grad_norm(vae.model.ss_decoder)
            
            # Calculate global step for logging
            global_step = epoch * len(train_dataloader) + batch_idx
            
            # Basic logging every batch
            visualizer.writer.add_scalar('Train/Loss', loss.item(), global_step)
            visualizer.writer.add_scalar('Train/GradNorm_Encoder', total_grad_norm_encoder, global_step)
            visualizer.writer.add_scalar('Train/GradNorm_Decoder', total_grad_norm_decoder, global_step)
            visualizer.writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            print(f"Epoch {epoch+1}/{cfg['epochs']}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.6f}, encoder_Total_grad_norm: {total_grad_norm_encoder:.6f}, decoder_Total_grad_norm: {total_grad_norm_decoder:.6f}")

            if batch_idx % cfg["log_interval"] == 0:
                acc = (pred_val>0.0).eq(gt_wnf_val>0.0).float().mean()
                loss_gt_wnf = loss_fn(gt_wnf_val,gt_sdf_val)
                acc_wnf = (gt_sdf_val>0.0).eq(gt_wnf_val>0.0).float().mean()
                
                # Create pred_sdf for visualization
                pred_sdf = torch.zeros_like(gt_sdf)
                pred_sdf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]] = pred_val
                
                # Use visualizer to save training batch results
                visualizer.save_training_batch_results(
                    epoch, batch_idx, loss.item(), acc.item(), feats_acc.item(),
                    total_grad_norm_encoder, total_grad_norm_decoder, global_step,
                    gt_sdf, pred_sdf, gt_wnf, pred_wnf_grad, masks,
                    vertices, faces, points, pred_normals, gt_normals, file_names=file_name
                )
                
                print(f"Epoch {epoch+1}/{cfg['epochs']}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.6f}, Acc: {acc.item():.6f}, Acc_wnf: {acc_wnf.item():.6f}, Loss_gt_wnf: {loss_gt_wnf.item():.6f}, Feats_acc: {feats_acc.item():.6f}")

        # Validation phase
        # val_loss, val_acc, val_acc_wnf, val_feats_acc = evaluate_model(
        #     vae, val_dataloader, device, cfg, loss_fn, epoch, 
        #     save_results=args.save_val_results and (epoch % 5 == 0 or epoch == cfg["epochs"] - 1)
        # )
        
        # Use visualizer to save validation results
        # visualizer.save_validation_epoch_results(epoch, val_loss, val_acc, val_acc_wnf, val_feats_acc)
        
        # print(f"\nValidation Results - Epoch {epoch+1}:")
        # print(f"Loss: {val_loss:.6f}, Acc: {val_acc:.6f}")
        # print(f"Acc_wnf: {val_acc_wnf:.6f}, Feats_acc: {val_feats_acc:.6f}")
        
        # # Save checkpoints using visualizer
        # is_best = val_loss < best_val_loss
        # if is_best:
        #     best_val_loss = val_loss
        #     best_val_acc = val_acc
        
        # visualizer.save_checkpoint(epoch, vae, optimizer, val_loss, val_acc, is_best=is_best)

    print("Training finished.")
    
    # Close visualizer and generate final report
    final_report = visualizer.close()
    
    print(f"🎉 Training completed successfully!")
    print(f"📊 Best validation loss: {final_report['best_performance']['best_val_loss']:.6f}")
    print(f"📊 Best validation accuracy: {final_report['best_performance']['best_val_accuracy']:.6f}")
    print(f"📁 Results saved to: {visualizer.output_dir}")





