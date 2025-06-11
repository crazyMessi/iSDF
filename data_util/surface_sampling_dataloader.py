import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
from data_util.dtype_utils import get_numpy_dtype, get_torch_dtype, numpy_to_tensor

# 加载配置
config = json.load(open("config/base_config.json"))

def mesh_normalize(vertices):
    """
    标准化网格顶点坐标
    """
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    m = np.max(np.sqrt(np.sum(vertices ** 2, axis=1)))
    vertices = vertices / m
    return vertices

class SurfaceSamplingDataset(Dataset):
    def __init__(self, data_path, num_samples=2048, normalize=True, transform=None, save_path=None):
        """
        在三角网格表面均匀采样点的数据集
        
        Args:
            data_path: 包含三角网格文件的目录路径
            num_samples: 在每个网格表面采样的点数
            normalize: 是否对网格进行标准化
            transform: 可选的数据变换
            save_path: 预处理数据的保存路径
        """
        self.data_path = data_path
        self.num_samples = num_samples
        self.normalize = normalize
        self.transform = transform
        
        # 获取所有网格文件
        self.file_list = []
        for ext in ['.ply', '.obj', '.off']:
            self.file_list.extend([f for f in os.listdir(data_path) if f.endswith(ext)])
        
        if len(self.file_list) == 0:
            raise ValueError(f"No mesh files found in {data_path}")
            
        print(f"Found {len(self.file_list)} mesh files")
        
        # 预处理数据的存储
        self.sampled_points_list = []
        self.sampled_normals_list = []
        self.vertices_list = []
        self.faces_list = []
        
        self._process_data(save_path)
        
    def _process_data(self, save_path):
        """
        预处理所有网格数据
        """
        # 检查是否有已保存的处理结果
        if save_path is not None:
            if os.path.exists(save_path):
                if os.path.exists(save_path + "sampled_points.npy"):
                    print("Loading pre-processed data from", save_path)
                    numpy_dtype = get_numpy_dtype()
                    self.sampled_points_list = np.load(save_path + "sampled_points.npy", allow_pickle=True)
                    self.sampled_points_list = [p.astype(numpy_dtype) for p in self.sampled_points_list]
                    self.sampled_normals_list = np.load(save_path + "sampled_normals.npy", allow_pickle=True)
                    self.sampled_normals_list = [n.astype(numpy_dtype) for n in self.sampled_normals_list]
                    self.vertices_list = np.load(save_path + "vertices.npy", allow_pickle=True)
                    self.vertices_list = [v.astype(numpy_dtype) for v in self.vertices_list]
                    self.faces_list = np.load(save_path + "faces.npy", allow_pickle=True)
                    self.faces_list = [f.astype(np.int32) for f in self.faces_list]
                    return
            else:
                os.makedirs(save_path, exist_ok=True)
                print(f"Created directory {save_path} for saving processed data")
        
        # 处理所有网格文件
        print("Processing mesh files...")
        numpy_dtype = get_numpy_dtype()
        
        for i in tqdm(range(len(self.file_list)), desc="Processing meshes", ncols=100):
            file_path = os.path.join(self.data_path, self.file_list[i])
            
            try:
                # 加载三角网格
                mesh = o3d.io.read_triangle_mesh(file_path)
                
                if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                    print(f"Warning: Empty mesh in {self.file_list[i]}, skipping...")
                    continue
                
                # 确保网格是有效的
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                # 获取顶点和面片
                vertices = np.asarray(mesh.vertices, dtype=numpy_dtype)
                faces = np.asarray(mesh.triangles, dtype=np.int32)
                
                # 标准化网格
                if self.normalize:
                    vertices = mesh_normalize(vertices)
                    mesh.vertices = o3d.utility.Vector3dVector(vertices)
                
                # 计算顶点法向量
                mesh.compute_vertex_normals()
                
                # 在网格表面均匀采样点
                sampled_pcd = mesh.sample_points_uniformly(number_of_points=self.num_samples)
                
                # 获取采样点和法向量
                sampled_points = np.asarray(sampled_pcd.points, dtype=numpy_dtype)
                sampled_normals = np.asarray(sampled_pcd.normals, dtype=numpy_dtype)
                
                # 如果法向量为空，重新计算
                if len(sampled_normals) == 0:
                    sampled_pcd.estimate_normals()
                    sampled_normals = np.asarray(sampled_pcd.normals, dtype=numpy_dtype)
                
                # 存储处理后的数据
                self.sampled_points_list.append(sampled_points)
                self.sampled_normals_list.append(sampled_normals)
                self.vertices_list.append(vertices)
                self.faces_list.append(faces)
                
            except Exception as e:
                print(f"Error processing {self.file_list[i]}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(self.sampled_points_list)} meshes")
        
        # 保存处理好的数据
        if save_path is not None:
            print(f"Saving processed data to {save_path}")
            np.save(save_path + "sampled_points.npy", np.array(self.sampled_points_list, dtype=object))
            np.save(save_path + "sampled_normals.npy", np.array(self.sampled_normals_list, dtype=object))
            np.save(save_path + "vertices.npy", np.array(self.vertices_list, dtype=object))
            np.save(save_path + "faces.npy", np.array(self.faces_list, dtype=object))
            print("Data saved successfully!")
    
    def __len__(self):
        return len(self.sampled_points_list)
    
    def __getitem__(self, idx):
        """
        返回采样点云、法向量和三角网格
        
        Returns:
            sampled_points: 在网格表面采样的点 [num_samples, 3]
            sampled_normals: 采样点的法向量 [num_samples, 3]
            vertices: 原始网格顶点 [num_vertices, 3]
            faces: 三角面片索引 [num_faces, 3]
        """
        sampled_points = self.sampled_points_list[idx].copy()
        sampled_normals = self.sampled_normals_list[idx].copy()
        vertices = self.vertices_list[idx].copy()
        faces = self.faces_list[idx].copy()
        
        # 应用变换（如果有）
        if self.transform:
            sampled_points, sampled_normals, vertices, faces = self.transform(
                sampled_points, sampled_normals, vertices, faces
            )
        
        return sampled_points, sampled_normals, vertices, faces


    @staticmethod
    def my_collate(batch):
        """
        自定义数据加载器
        """
        points = [item[0] for item in batch]
        normals = [item[1] for item in batch]
        vertices = [item[2] for item in batch]
        faces = [item[3] for item in batch]
        
        points = torch.stack(points, dim=0)
        normals = torch.stack(normals, dim=0)
        return points, normals, vertices, faces


class SurfaceSamplingDatasetTorch(SurfaceSamplingDataset):
    """
    返回PyTorch张量的版本
    """
    def __init__(self, data_path, num_samples=2048, normalize=True, transform=None, 
                 save_path=None, device='cpu'):
        super().__init__(data_path, num_samples, normalize, transform, save_path)
        self.device = device
        
    def __getitem__(self, idx):
        sampled_points, sampled_normals, vertices, faces = super().__getitem__(idx)
        
        # 转换为PyTorch张量
        torch_dtype = get_torch_dtype()
        sampled_points = torch.tensor(sampled_points, dtype=torch_dtype, device=self.device)
        sampled_normals = torch.tensor(sampled_normals, dtype=torch_dtype, device=self.device)
        vertices = torch.tensor(vertices, dtype=torch_dtype, device=self.device)
        faces = torch.tensor(faces, dtype=torch.long, device=self.device)
        
        return sampled_points, sampled_normals, vertices, faces



def get_surface_sampling_dataloader(data_path, num_samples=2048, batch_size=8, 
                                  shuffle=True, num_workers=4, normalize=True,
                                  save_path=None, use_torch=True, device='cpu'):
    """
    创建表面采样数据加载器
    
    Args:
        data_path: 网格文件目录
        num_samples: 每个网格表面采样点数
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数
        normalize: 是否标准化网格
        save_path: 预处理数据保存路径
        use_torch: 是否返回PyTorch张量
        device: 设备类型
        
    Returns:
        DataLoader对象
    """
    if use_torch:
        dataset = SurfaceSamplingDatasetTorch(
            data_path=data_path,
            num_samples=num_samples,
            normalize=normalize,
            save_path=save_path,
            device=device
        )
    else:
        
        dataset = SurfaceSamplingDataset(
            data_path=data_path,
            num_samples=num_samples,
            normalize=normalize,
            save_path=save_path
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if not use_torch else 0,  # PyTorch张量版本不使用多进程
        collate_fn=SurfaceSamplingDataset.my_collate
    )
    
    return dataloader


# 示例用法
if __name__ == "__main__":
    # 示例：如何使用数据加载器
    save_path = "/mnt/lizd/workdata/iSDF/mesh_segment/temp/"
    data_path = "/mnt/lizd/workdata/iSDF/mesh_segment/"
    # 创建数据加载器
    dataloader = get_surface_sampling_dataloader(
        data_path=data_path,
        num_samples=2048,
        batch_size=4,
        save_path=save_path,
        use_torch=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 测试数据加载
    for batch_idx, (points, normals, vertices, faces) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Sampled points shape: {points.shape}")
        print(f"  Sampled normals shape: {normals.shape}")
        
        
        if batch_idx >= 2:  # 只测试前3个batch
            break 