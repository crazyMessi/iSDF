import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import tools
import os
from scipy.spatial import cKDTree
from tqdm import tqdm  # 导入tqdm进度条库
import json
from dtype_utils import get_numpy_dtype, get_torch_dtype, numpy_to_tensor



config = json.load(open("config/base_config.json"))
PTS_COUNT = config["PTS_COUNT"]

class PointCloudDataset(Dataset):    
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path: Path to the directory containing point cloud files
            transform: Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.transform = transform
        self.file_list = [] 
        self.file_list = os.listdir(self.data_path)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load point cloud file
        pcd = o3d.io.read_point_cloud(self.data_path + self.file_list[idx])
        points = np.asarray(pcd.points, dtype=get_numpy_dtype())
        normals = np.asarray(pcd.normals, dtype=get_numpy_dtype())
        # if self.transform:
        return points, normals
    
class TriangleMeshDataset(Dataset):
    def __init__(self, data_path,transform=None):
        self.data_path = data_path
        self.transform = transform
        self.file_list = [] 
        self.file_list = os.listdir(self.data_path)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load 
        mesh = o3d.io.read_triangle_mesh(self.data_path + self.file_list[idx])
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices, dtype=get_numpy_dtype())
        faces = np.asarray(mesh.faces, dtype=get_numpy_dtype())
        normals = np.asarray(mesh.vertex_normals, dtype=get_numpy_dtype())
        return vertices, normals, faces



'''
QueryPointsDataset
* 每次返回point、normal, 根据point得到的mask
'''
class QueryPointsDataset(Dataset):
    def __init__(self,pointcloudDataset,resolution,k,device,save_path=None):
        if not(type(pointcloudDataset) == PointCloudDataset or type(pointcloudDataset) == TriangleMeshDataset):
            assert False
        self.fileDataset = pointcloudDataset
        bbox = np.array([[-1,1],[-1,1],[-1,1]])
        pts, grid_shape = tools.create_uniform_grid(bbox=bbox,resolution=resolution)
        self.bbox = bbox
        self.resolution = resolution
        self.pts = pts
        self.grid_shape = grid_shape
        self.kdtree = cKDTree(pts)
        # self.masks = [] # 太大了 每次都重新计算吧
        # self.distances = []
        self.k = k
        self.device = device
        self.point_list = [] # 变化后的每个点云
        self.normal_list = []
        self.faces_list = []
        self._iXForm = [] # 每个点云的逆变换矩阵
        # self.idx_list = [] # 每个点云的索引
        self._process_data(save_path)
        
    def _process_data(self,save_path):
        # 首先检查是否有已保存的处理结果
        if save_path is not None:
            if os.path.exists(save_path):
                if os.path.exists(save_path + "point_list.npy"):
                    print("Loading pre-processed data from", save_path)
                    # 明确指定dtype，并确保加载后转换为正确的类型
                    numpy_dtype = get_numpy_dtype()
                    self.point_list = np.load(save_path + "point_list.npy", allow_pickle=True)
                    self.point_list = [p.astype(numpy_dtype) for p in self.point_list]
                    self.normal_list = np.load(save_path + "normal_list.npy", allow_pickle=True)
                    self.normal_list = [n.astype(numpy_dtype) for n in self.normal_list]
                    self._iXForm = np.load(save_path + "iXForm.npy", allow_pickle=True)
                    self._iXForm = [t.astype(numpy_dtype) for t in self._iXForm]
                    if type(self.fileDataset) == TriangleMeshDataset:
                        self.faces_list = np.load(save_path + "faces_list.npy", allow_pickle=True)
                        self.faces_list = [f.astype(numpy_dtype) for f in self.faces_list]
                    return
            else:
                dirname = os.path.dirname(save_path)
                tools.rmkdir(dirname)
                print(f"Created directory {save_path} for saving processed data")
                
        # 如果没有预处理的数据，开始处理点云
        print("Processing point clouds...")
        total_points = len(self.fileDataset)
        
        numpy_dtype = get_numpy_dtype()
        idx_list = []
        # 使用tqdm创建进度条
        for i in tqdm(range(total_points), desc="Processing point clouds", ncols=100):
            if type(self.fileDataset) == PointCloudDataset:
                points, normals = self.fileDataset[i]
            else:
                points, normals, faces = self.fileDataset[i]
       
            
            # 确保数据类型一致
            points = points.astype(numpy_dtype)
            normals = normals.astype(numpy_dtype)
            
            # 随机选择PTS_COUNT个点
            idx = np.random.choice(points.shape[0], PTS_COUNT, replace=False)
            points = points[idx]
            normals = normals[idx]
            
            points, iXForm = tools.transform_points(points)
            
            # 确保数据类型一致
            points = points.astype(numpy_dtype)
            iXForm = iXForm.astype(numpy_dtype)
            
            # 计算查询点的距离和索引
            dists, idxs = self.kdtree.query(points, k=self.k)
            
            # 创建掩码
            mask = np.zeros(self.pts.shape[0], dtype=numpy_dtype)
            mask[idxs] = 1
            # 保存结果
            # self.masks.append(mask)
            self.point_list.append(points)
            self.normal_list.append(normals)
            self._iXForm.append(iXForm)
            if type(self.fileDataset) == TriangleMeshDataset:
                self.faces_list.append(faces)
            idx_list.append(idx)
            
        # 如果提供了保存路径，则保存处理好的数据
        if save_path is not None:
            print(f"Saving processed data to {save_path}")
            np.save(save_path + "point_list.npy", np.array(self.point_list, dtype=object))
            np.save(save_path + "normal_list.npy", np.array(self.normal_list, dtype=object))
            np.save(save_path + "iXForm.npy", np.array(self._iXForm, dtype=object))
            if type(self.fileDataset) == TriangleMeshDataset:
                np.save(save_path + "faces_list.npy", np.array(self.faces_list, dtype=object))
            np.save(save_path + "idx_list.npy", np.array(idx_list, dtype=object))
            print("Data saved successfully!")



        return
        
    def __len__(self):
        return len(self.point_list)
    
    def __getitem__(self, idx):
        # 在这里将数据转移到指定设备上
        points = self.point_list[idx]
        normals = self.normal_list[idx] 
        if type(self.fileDataset) == TriangleMeshDataset:
            faces = self.faces_list[idx]
        # 计算查询点的距离和索引
        dists, idxs = self.kdtree.query(points, k=self.k)
        
        # 创建掩码
        mask = np.zeros(self.pts.shape[0], dtype=get_numpy_dtype())
        mask[idxs] = 1
        
        # # 转换为torch张量，并确保数据类型一致
        # torch_dtype = get_torch_dtype()
        # points = torch.tensor(points, dtype=torch_dtype).to(self.device)
        # normals = torch.tensor(normals, dtype=torch_dtype).to(self.device)
        # if type(self.fileDataset) == TriangleMeshDataset:
        #     faces = torch.tensor(self.faces_list[idx], dtype=torch_dtype).to(self.device)
        # mask = torch.tensor(mask, dtype=torch_dtype).to(self.device)

        if type(self.fileDataset) == TriangleMeshDataset:
            return points, normals, mask, faces
        else:
            return points, normals, mask




class QueryPointsDatasetMix(Dataset):
    
    def __init__(self,query_points_dataset,save_path=None,device='cpu'):
        self.query_points = []
        self.sdistane = []
        self.device = device
        if save_path is not None:
            if os.path.exists(save_path):
                try:
                    if os.path.exists(save_path + "query_points.npy") and os.path.exists(save_path + "sdistane.npy"):
                        print("Loading pre-processed data from", save_path)
                        numpy_dtype = get_numpy_dtype()
                        self.query_points = np.load(save_path + "query_points.npy", allow_pickle=True)
                        self.query_points = [p.astype(numpy_dtype) for p in self.query_points]
                        self.query_points = np.concatenate(self.query_points,axis=0)
                        self.sdistane = np.load(save_path + "sdistane.npy", allow_pickle=True)
                        self.sdistane = [d.astype(numpy_dtype) for d in self.sdistane]
                        self.sdistane = np.concatenate(self.sdistane,axis=0)
                        return
                except:
                    self.query_points = []
                    self.sdistane = []  
                    print("No pre-processed data found in", save_path)
            else:
                dirname = os.path.dirname(save_path)
                tools.rmkdir(dirname)
                print(f"Created directory {save_path} for saving processed data")

        pts, _ = tools.create_uniform_grid(resolution=query_points_dataset.resolution)

        for i in range(len(query_points_dataset)):
            points, normals, mask = query_points_dataset[i]
            qp = pts[mask==1]   
            self.query_points.append(np.concatenate((qp,np.ones((qp.shape[0],1))*i),axis=1))
            tree = cKDTree(points)
            dists, idxs = tree.query(qp, k=1)
            npoint = points[idxs]
            displacement = npoint - qp
            nnormal = normals[idxs]
            sign = np.sum(nnormal * displacement, axis=1) < 0 
            dists[sign<0] = -dists[sign<0]
            self.sdistane.append(dists)
            
        
        
        self.query_points = np.concatenate(self.query_points,axis=0)
        self.sdistane = np.concatenate(self.sdistane,axis=0)
        
        if save_path is not None:
            np.save(save_path + "query_points.npy",self.query_points)
            np.save(save_path + "sdistane.npy",self.sdistane)
            
    def __len__(self):
        return len(self.query_points)
    
    def __getitem__(self, idx):
        torch_dtype = get_torch_dtype()
        query_points = self.query_points[idx]
        sdistane = self.sdistane[idx]
        return query_points, sdistane
        
     

def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4):
    dataset = PointCloudDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader 