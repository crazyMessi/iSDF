import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from data_util.dtype_utils import get_numpy_dtype, get_torch_dtype, tensor_to_numpy
from tools import poission_rec
import pysdf
import trimesh
from scipy.spatial import cKDTree


class MeshSDF:
    def __init__(self, vertices, faces):
        """
        初始化 MeshSDF 类
        
        参数:
            vertices: np.ndarray, 形状为 (N, 3) 的顶点坐标数组
            faces: np.ndarray, 形状为 (M, 3) 的面索引数组
        """
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        
        # 创建trimesh对象用于法向量计算
        self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        
        # 构建 AABB 树，这里用 trimesh 的实现
        self.tree = trimesh.proximity.ProximityQuery(self.mesh)
        
        # 计算每个面的法向量
        self.face_normals = self.mesh.face_normals
        
        # 创建面的质心 KDTree 以加速最近面查询
        self.face_centers = self.mesh.triangles_center
        self.face_tree = cKDTree(self.face_centers)
        
        # 检查网格是否封闭（可选）
        self.is_watertight = self.mesh.is_watertight
        # if not self.is_watertight:
            # print("警告: 网格不是封闭的，可能会导致SDF计算不准确")
    
    def query(self, points):
        """
        计算查询点到网格的有符号距离
        
        参数:
            points: np.ndarray, 形状为 (P, 3) 的查询点坐标数组
            
        返回:
            distances: np.ndarray, 形状为 (P,) 的有符号距离数组
        """
        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # 使用 AABB 树计算距离和最近点
        unsigned_distances, closest_points, face_ids = self.closest_points_and_faces(points)
        
        # 获取对应面的法向量
        normals = self.face_normals[face_ids]
        
        # 计算方向向量（从最近点到查询点）
        direction_vectors = points - closest_points
        
        # 单位化方向向量
        directions_norm = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
        # 避免除以零
        mask = directions_norm > 1e-10
        direction_vectors = np.where(mask, direction_vectors / directions_norm, direction_vectors)
        
        
        # 计算点积确定符号
        signs = np.sum(direction_vectors * normals, axis=1)
        signs = np.sign(signs)
        
        # 对于不封闭的网格，可以使用光线投射来确定内外（如果需要更准确）
        if not self.is_watertight:
            # 这里可以添加光线投射代码来更准确地确定内外
            pass
        
        # 计算有符号距离
        signed_distances = signs * unsigned_distances
        
        if len(signed_distances) == 1:
            return signed_distances[0]
        return signed_distances
    
    def accurate_closest_points_and_faces(self, points):
        """
        查找给定点到网格的最近点和对应的面
        
        参数:
            points: np.ndarray, 形状为 (P, 3) 的查询点坐标数组
            
        返回:
            distances: np.ndarray, 形状为 (P,) 的无符号距离数组
            closest_points: np.ndarray, 形状为 (P, 3) 的最近点坐标数组
            face_ids: np.ndarray, 形状为 (P,) 的最近面的索引数组
        """
        # # 使用 trimesh 的 closest_point 函数
        # distances, closest_points_idx = self.tree.vertex(points)
        # closest_points = self.vertices[closest_points_idx]
        
        # # 找到最近的面
        # _, face_ids = self.face_tree.query(closest_points)
        # 使用 trimesh 的精确距离计算
        # 这比仅仅使用顶点距离更准确
        accurate_closest_points,accurate_distances,face_ids = self.tree.on_surface(points)
        return accurate_distances, accurate_closest_points, face_ids
    
    def closest_points_and_faces(self, points):
        """
        计算查询点到网格的最近点和对应的面
        
        参数:
            points: np.ndarray, 形状为 (P, 3) 的查询点坐标数组
            
        返回:
            distances: np.ndarray, 形状为 (P,) 的无符号距离数组
            closest_points: np.ndarray, 形状为 (P, 3) 的最近点坐标数组
            face_ids: np.ndarray, 形状为 (P,) 的最近面的索引数组
        """
        # 使用 trimesh 的 closest_point 函数
        distances, closest_points_idx = self.tree.vertex(points)
        closest_points = self.vertices[closest_points_idx]
        
        # 找到最近的面
        _, face_ids = self.face_tree.query(closest_points)
        
        return distances, closest_points, face_ids
        
    
    def contains(self, points):
        """
        判断点是否在网格内部
        
        参数:
            points: np.ndarray, 形状为 (P, 3) 的查询点坐标数组
            
        返回:
            inside: np.ndarray, 形状为 (P,) 的布尔数组，表示每个点是否在内部
        """
        # if not self.is_watertight:
        #     print("警告: 网格不是封闭的，内部判断可能不准确")
        
        signed_distances = self.query(points)
        if np.isscalar(signed_distances):
            return signed_distances <= 0
        return signed_distances <= 0

    def batch_query(self, points, batch_size=1000):
        """
        批量计算查询点到网格的有符号距离，适用于大量点的情况
        
        参数:
            points: np.ndarray, 形状为 (P, 3) 的查询点坐标数组
            batch_size: int, 每批处理的点数
            
        返回:
            distances: np.ndarray, 形状为 (P,) 的有符号距离数组
        """
        points = np.array(points, dtype=np.float32)
        num_points = len(points)
        results = np.zeros(num_points)
        
        for i in range(0, num_points, batch_size):
            batch_points = points[i:i+batch_size]
            results[i:i+batch_size] = self.query(batch_points)
            
        return results

class SDFField:
    def __init__(self, resolution=64):
        self.resolution = resolution
    
    def compute_sdf(self, points, normals, voxel_grid):
        """
        Compute SDF values for a voxel grid given oriented point cloud
        Args:
            points: N x 3 tensor of points
            normals: N x 3 tensor of normals
            voxel_grid: M x 3 tensor of voxel centers
        Returns:
            sdf_values: M tensor of SDF values
        """
        # 获取数据类型
        numpy_dtype = get_numpy_dtype()
        torch_dtype = get_torch_dtype()
        
        # Convert to numpy for KDTree
        points_np = tensor_to_numpy(points)
        normals_np = tensor_to_numpy(normals)
        if isinstance(voxel_grid,torch.Tensor):
            voxel_grid_np = tensor_to_numpy(voxel_grid)
        else:
            voxel_grid_np = voxel_grid
        
        # Build KD-tree
        tree = KDTree(points_np)
        
        # Query KD-tree for nearest neighbors
        distances, indices = tree.query(voxel_grid_np, k=1)
        distances = distances.flatten()
        distances = np.sqrt(distances)
        
        # Get closest points and their normals
        closest_points = points_np[indices.flatten()]
        closest_normals = normals_np[indices.flatten()]
        
        # Compute vectors from closest points to voxels
        vectors = voxel_grid - closest_points
        
        # Compute signed distances: dot product of vectors with normals
        sign = np.sum(vectors * closest_normals, axis=1)
        sign = np.sign(sign)
        sign[sign == 0] = 1 # 0 取 1 即垂直于法向量的点视为在外部
        
        
        # Convert back to torch tensor with specified dtype
        signed_distances = sign * distances
        signed_distances = torch.tensor(signed_distances, dtype=torch_dtype).to(points.device)
        
        return signed_distances

    def create_voxel_grid(self, points, padding=0.1):
        """
        Create a uniform voxel grid covering the point cloud
        Args:
            points: N x 3 tensor of points
            padding: padding around the point cloud
        Returns:
            voxel_grid: M x 3 tensor of voxel centers
        """
        # Get bounding box
        min_coords = torch.min(points, dim=0)[0] - padding
        max_coords = torch.max(points, dim=0)[0] + padding
        
        # Create uniform grid
        x = torch.linspace(min_coords[0], max_coords[0], self.resolution)
        y = torch.linspace(min_coords[1], max_coords[1], self.resolution)
        z = torch.linspace(min_coords[2], max_coords[2], self.resolution)
        
        X, Y, Z = torch.meshgrid(x, y, z)
        voxel_grid = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        
        return voxel_grid 

    def compute_mesh_sdf(self, vertices, faces, points=None, n_threads=None):
        """
        Compute SDF values for given points using mesh data
        Args:
            vertices: N x 3 array/tensor of mesh vertices
            faces: M x 3 array/tensor of mesh faces (indices)
            points: P x 3 array/tensor of query points (optional)
            n_threads: Number of threads to use for computation (optional)
        Returns:
            If points is provided: P tensor of SDF values
            Otherwise: SDF function that can be called on points
        """
        # Convert to numpy if tensors
        if isinstance(vertices, torch.Tensor):
            vertices = tensor_to_numpy(vertices)
        if isinstance(faces, torch.Tensor):
            faces = tensor_to_numpy(faces)
        
        # Create SDF object from pysdf
        self.mesh_sdf = pysdf.SDF(vertices, faces)
        
        # If points are provided, compute SDF values immediately
        if points is not None:
            if isinstance(points, torch.Tensor):
                points_np = tensor_to_numpy(points)
                device = points.device
                dtype = points.dtype
            else:
                points_np = points
                device = torch.device('cpu')
                dtype = get_torch_dtype()
                
            if n_threads is not None:
                sdf_values = self.mesh_sdf(points_np, n_threads=n_threads)
            else:
                sdf_values = self.mesh_sdf(points_np)
                
            return torch.tensor(sdf_values, dtype=dtype).to(device)
        
        # Otherwise return the mesh_sdf object itself, which can be called later
        return self.mesh_sdf
    
    def contains(self, points, n_threads=None):
        """
        Check if points are contained in the mesh
        Args:
            points: P x 3 array/tensor of query points
            n_threads: Number of threads to use for computation (optional)
        Returns:
            P boolean tensor indicating whether each point is contained
        """
        if self.mesh_sdf is None:
            raise ValueError("Must call compute_mesh_sdf first to initialize mesh SDF")
            
        if isinstance(points, torch.Tensor):
            points_np = tensor_to_numpy(points)
            device = points.device
        else:
            points_np = points
            device = torch.device('cpu')
            
        if n_threads is not None:
            contained = self.mesh_sdf.contains(points_np, n_threads=n_threads)
        else:
            contained = self.mesh_sdf.contains(points_np)
            
        return torch.tensor(contained, dtype=torch.bool).to(device)
    
    def nearest_points(self, points, n_threads=None):
        """
        Find nearest points on mesh surface
        Args:
            points: P x 3 array/tensor of query points
            n_threads: Number of threads to use for computation (optional)
        Returns:
            P x 3 tensor of nearest points on the mesh surface
        """
        if self.mesh_sdf is None:
            raise ValueError("Must call compute_mesh_sdf first to initialize mesh SDF")
            
        if isinstance(points, torch.Tensor):
            points_np = tensor_to_numpy(points)
            device = points.device
            dtype = points.dtype
        else:
            points_np = points
            device = torch.device('cpu')
            dtype = get_torch_dtype()
            
        if n_threads is not None:
            nearest = self.mesh_sdf.nn(points_np, n_threads=n_threads)
        else:
            nearest = self.mesh_sdf.nn(points_np)
            
        return torch.tensor(nearest, dtype=dtype).to(device)
    
    def sample_surface_points(self, n_points, n_threads=None):
        """
        Sample uniform points on the mesh surface
        Args:
            n_points: Number of points to sample
            n_threads: Number of threads to use for computation (optional)
        Returns:
            n_points x 3 tensor of points on the mesh surface
        """
        if self.mesh_sdf is None:
            raise ValueError("Must call compute_mesh_sdf first to initialize mesh SDF")
            
        if n_threads is not None:
            points = self.mesh_sdf.sample_surface(n_points, n_threads=n_threads)
        else:
            points = self.mesh_sdf.sample_surface(n_points)
            
        return torch.tensor(points, dtype=get_torch_dtype())
    
    @property
    def surface_area(self):
        """
        Get the surface area of the mesh
        Returns:
            Surface area as a scalar
        """
        if self.mesh_sdf is None:
            raise ValueError("Must call compute_mesh_sdf first to initialize mesh SDF")
            
        return self.mesh_sdf.surface_area 