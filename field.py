import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from dtype_utils import get_numpy_dtype, get_torch_dtype, tensor_to_numpy
from tools import poission_rec
import pysdf


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
        
        # Build KD-tree
        tree = KDTree(points_np)
        
        # Query KD-tree for nearest neighbors
        distances, indices = tree.query(voxel_grid, k=1)
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