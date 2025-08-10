import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
import shutil
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
import cal_wnf
import update_normal
import open3d as o3d

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

class VoxelGridVAEBuilder(torch.nn.Module):
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

def load_model(checkpoint_path, device):
    """加载预训练的模型"""
    print(f"Loading model from {checkpoint_path}")
    model = VoxelGridVAEBuilder().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()    
    return model


def main():
    parser = argparse.ArgumentParser(description='Run iSDF normal estimation')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum number of iterations')
    args = parser.parse_args()

    # 加载配置
    cfg = get_config(dataset_name=args.dataset_name)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    dataloader = get_surface_sampling_dataloader(
        data_path=cfg["data_path"],
        num_samples=cfg["num_samples"],
        batch_size=2,  # 每次处理一个模型
        save_path=cfg["save_path"],
        use_torch=cfg["use_torch_dataloader"],
        device=device,
        normalize=True,
        shuffle=False,
        val_split=0.0
    )

    # 加载模型
    model = load_model(args.checkpoint, device)

    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{cfg.get('experiment_output_path')}/isdf_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)


    # 处理每个模型
    for batch_idx, (points, gt_normals, vertices, faces, filename) in enumerate(dataloader):
        print(f"\nProcessing model {batch_idx + 1}/{len(dataloader)}")
        
        points = points.to(device)
        voxel_coords = points.clone()
        assert points.max() <= 1.0 and points.min() >= -1.0, "Points should be normalized to [-1, 1] range"
        voxel_coords = ((voxel_coords + 1.0) / 2.0 * cfg["r"]).long()
        B,N,_ = voxel_coords.shape
        batch_indices = torch.arange(B, device=voxel_coords.device).repeat_interleave(N).unsqueeze(-1)  # [N*B, 1]
        voxel_coords = torch.cat([batch_indices, voxel_coords.reshape(-1, 3)], dim=-1).int()  # [N*B, 4]
        
        model.eval()
        
        # updated_normal = 
        i_resolu = cfg["r"]
        o_resolu = cfg["R"]
        pred_normals = torch.zeros_like(points, dtype=torch.float32, device=device)
        voxel_center, grid_shape = tools.create_uniform_grid(i_resolu, bbox=np.array([[-1,1],[-1,1],[-1,1]]))
        feat_shape = [B] + list(grid_shape)
        gt_wnf = torch.zeros(feat_shape,device=device)
        pred_wnf = torch.zeros(feat_shape,device=device)
        gt_sdf = torch.zeros(feat_shape,device=device)
        pred_wnf_grad = torch.zeros(feat_shape,device=device)
        masks = torch.zeros(feat_shape,device=device)
         
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
            
            # 计算wnf
            gt_wnf_i = cal_wnf.compute_winding_number_torch_api(points[i],gt_normals[i],query_points_i,epsilon=1e-8,batch_size=10000)
            gt_wnf[i][mask_cuda] = gt_wnf_i
            
            # 计算sdf
            mesh_sdf = MeshSDF(vertices[i].cpu().numpy(), faces[i].cpu().numpy())
            sdf_values = mesh_sdf.query(query_points_i.cpu().numpy())
            gt_sdf[i][mask_cuda] = torch.from_numpy(sdf_values).to(device,dtype=torch.float32)
            
        indices = torch.nonzero(masks).int()
        gt_sdf_val = gt_sdf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
        wnf_val = pred_wnf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
        pred_wnf_grad_val = pred_wnf_grad[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
        gt_wnf_val = gt_wnf[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]
        wnf_val = torch.tanh(wnf_val)
        gt_wnf_val = torch.tanh(gt_wnf_val)
        pred_wnf_grad_val = torch.tanh(pred_wnf_grad_val)
        
        pred_normals = estimate_normals_pca(points, k=cfg["pca_knn"]) # (B, N_points, 3)
        for i in range(10):            
            for j in range(B):
                mask = masks[j].cpu().numpy()>0
                query_points_i = torch.from_numpy(voxel_center[mask.flatten()]).to(device,dtype=torch.float32)
                wnf_i = cal_wnf.compute_winding_number_torch_api(points[j], pred_normals[j], query_points_i, epsilon=1e-8, batch_size=10000)
                pred_wnf[j][mask] = wnf_i
                wnf_grad = tools.compute_gradient(pred_wnf[j], mask)
                pred_wnf_grad[j][mask] = wnf_grad[mask]

            feats = pred_wnf_grad_val.unsqueeze(1)
            sp_feats = sp.SparseTensor(feats, indices)
            sp_feats = model(sp_feats)
            pred_val = sp_feats.feats.detach().cpu().numpy()
            
            loss_fn = CustomLoss()
            loss = loss_fn(torch.from_numpy(pred_val.squeeze()).to(device), gt_wnf_val)
            print(f"Iteration {i+1}/{args.max_iterations}, Loss: {loss.item()}")
            field = np.zeros(feat_shape, dtype=float)
            field[masks.cpu().numpy()>0] = pred_val.squeeze()
            
            

            for j in range(B):
                v, f = tools.extract_surface_from_scalar_field(field[j], level=0, mask=mask, resolution=cfg["R"])
                # TODO 这里重复建立kdtree
                normals = update_normal.compute_point_normals_from_mesh(points[j].cpu().numpy(), v, f,20)
                pred_normals[j] = torch.from_numpy(normals).to(device, dtype=torch.float32)

                # === 结果保存 ===
                result_dir = output_dir / f"model_{filename[j]}"
                suffice = f"_iter_{i}" if i > 0 else ""
                result_dir.mkdir(parents=True, exist_ok=True)
                # 使用o3d保存点云和法向量
                op = o3d.geometry.PointCloud()
                op.points = o3d.utility.Vector3dVector(points[j].cpu().numpy())
                op.normals = o3d.utility.Vector3dVector(pred_normals[j].cpu().numpy())
                o3d.io.write_point_cloud(str(result_dir / f"points_normals_{suffice}.ply"), op)
                
                
                # 使用trimesh保存表面
                if v is not None and f is not None:
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(v)
                    mesh.triangles = o3d.utility.Vector3iVector(f)
                    o3d.io.write_triangle_mesh(str(result_dir / f"surface_{suffice}.ply"), mesh)
                    
                # 保存scalar field
                np.save(str(result_dir / f"scalar_field_{suffice}.npy"), field[j])
                
                acc = tools.cal_normal_acc(gt_normals[j].cpu().numpy(), pred_normals[j].cpu().numpy())
                # 保存日志(追加写入)
                with open(str(result_dir / "info.txt"), 'a') as f_log:
                    f_log.write(f"iter: acc: {acc}\n")
            
                if i == 0:
                    # 保存gt
                    op_gt = o3d.geometry.PointCloud()
                    op_gt.points = o3d.utility.Vector3dVector(points[j].cpu().numpy())
                    op_gt.normals = o3d.utility.Vector3dVector(gt_normals[j].cpu().numpy())
                    o3d.io.write_point_cloud(str(result_dir / f"points_norm als_gt.ply"), op_gt)    
                    # 保存mesh
                    mesh_gt = o3d.geometry.TriangleMesh()
                    mesh_gt.vertices = o3d.utility.Vector3dVector(vertices[j].cpu().numpy())
                    mesh_gt.triangles = o3d.utility.Vector3iVector(faces[j].cpu().numpy())
                    o3d.io.write_triangle_mesh(str(result_dir / f"mesh_gt.ply"), mesh_gt) 

        
                
                

if __name__ == "__main__":
    main()
