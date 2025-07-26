import numpy as np
from skimage import measure
import open3d as o3d
from scipy.spatial import cKDTree
from data_util.dtype_utils import get_numpy_dtype

# 0. 计算指定分辨率的 winding number 场
def normalize_points(points):
    """
    将点云归一化到单位立方体 [0, 1]^3 内
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    scale = (max_coords - min_coords) * 1.1
    normalized = (points - min_coords) / scale
    return normalized, min_coords, scale

# 1. 将点云变换到单位立方体 [-1, 1]^3 内
def transform_points(points):
    numpy_dtype = get_numpy_dtype()
    Move = np.eye(4, dtype=numpy_dtype)
    Scale = np.eye(4, dtype=numpy_dtype)
    XForm = np.eye(4, dtype=numpy_dtype)
    center = np.mean(points, axis=0).astype(numpy_dtype)
    t = (np.max(points, axis=0) - np.min(points, axis=0)).astype(numpy_dtype)
    scale = np.array([t.max(), t.max(), t.max()], dtype=numpy_dtype)
    Move[:3, 3] = -center
    Scale[[0, 1, 2], [0, 1, 2]] = 0.9/scale
    XForm = Scale @ Move
    tpoints = np.concatenate((points, np.ones((points.shape[0], 1), dtype=numpy_dtype)), axis=1)
    tpoints = (XForm @ tpoints.T).T
    iXForm = np.linalg.inv(XForm)
    return tpoints[:, :3].astype(numpy_dtype), iXForm.astype(numpy_dtype)


def visualize_partition(segmented_grid, partition_labels):
    """
    将超体素的二分结果映射回原始体素并可视化
    参数:
      segmented_grid: 原始的超体素标签网格
      partition_labels: 二分后每个超体素的标签(0或1)
    """
    new_grid = np.zeros_like(segmented_grid)
    dict_label = {label: partition_labels[label] for label in np.unique(segmented_grid)}
    f = np.vectorize(lambda x: dict_label[x])
    new_grid = f(segmented_grid)
    return new_grid

import pymeshlab

def pymeshlab_normal_estimate(points,k):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(points))
    ms.compute_normal_for_point_clouds(k=k)
    normals = ms.current_mesh().vertex_normal_matrix()
    return np.asarray(normals)

def PCA_normal_estimate(points):
    """
    使用PCA估计点云的法向量
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(1)
    return np.asarray(pcd.normals)

def random_normal_estimate(points):
    """
    随机估计点云的法向量
    """
    normals = np.random.rand(points.shape[0],3)
    while(np.any(np.linalg.norm(normals,axis=1)<0.01)):
        normals = np.random.rand(points.shape[0],3)
    normals = normals / np.linalg.norm(normals,axis=1,keepdims=True)
    return normals


# 可视化示例代码
def plot_partition(partition_grid, save_path="./temp/partition.png",slice_idx=None,bbox=None,points=None,vis=False):
    """
    可视化二分结果
    参数:
      partition_grid: 二分结果网格
    """
    import matplotlib.pyplot as plt
    rmkdir("/".join(save_path.split("/")[0:-1]))
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    if slice_idx is None:
        # 选择中间的切片进行显示
        z_slice = partition_grid.shape[2] // 2
        y_slice = partition_grid.shape[1] // 2
        x_slice = partition_grid.shape[0] // 2
    else:
        z_slice = slice_idx[0]
        y_slice = slice_idx[1]
        x_slice = slice_idx[2]
    
    def scatter_points(axis,points,bbox,ax,slice_idx):
        step = (bbox[axis,1]-bbox[axis,0])/(partition_grid.shape[axis]-1)
        _range = (bbox[axis,0] + step * slice_idx,bbox[axis,0] + step * (slice_idx+1))
        points_in_slice = points[(points[:,axis] >= _range[0]) & (points[:,axis] <= _range[1])]
        x = (axis + 1) % 3
        y = (axis + 2) % 3
        if x < y:
            x,y = y,x
        # 映射到0~resolution-1
        points_in_slice[:,x] = (points_in_slice[:,x] - bbox[x,0]) / (bbox[x,1] - bbox[x,0]) * (partition_grid.shape[x]-1)
        points_in_slice[:,y] = (points_in_slice[:,y] - bbox[y,0]) / (bbox[y,1] - bbox[y,0]) * (partition_grid.shape[y]-1)
        ax.scatter(points_in_slice[:,x],points_in_slice[:,y],c='r',s=1)
    
    
    # XY平面
    ax1 = fig.add_subplot(131)
    if points is not None and bbox is not None:
        scatter_points(2,points,bbox,ax1,z_slice)
    ax1.imshow(partition_grid[:, :, z_slice], cmap='coolwarm',origin='lower')
    ax1.set_title('XY Plane (z=%d)' % z_slice)

    # XZ平面
    ax2 = fig.add_subplot(132)
    ax2.imshow(partition_grid[:, y_slice, :], cmap='coolwarm',origin='lower')
    ax2.set_title('XZ Plane (y=%d)' % y_slice)
    if points is not None and bbox is not None:
        scatter_points(1,points,bbox,ax2,y_slice)
    
    # YZ平面
    ax3 = fig.add_subplot(133)
    ax3.imshow(partition_grid[x_slice, :, :], cmap='coolwarm',origin='lower')
    ax3.set_title('YZ Plane (x=%d)' % x_slice)
    if points is not None and bbox is not None:
        scatter_points(0,points,bbox,ax3,x_slice)

    plt.tight_layout()
    
    plt.savefig(save_path)
    if vis:
        plt.show()
    plt.close()

import os

# 递归创建文件夹
def rmkdir(path):
    if path == '':
        return
    parent = os.path.dirname(path)
    if parent == '':
        if not os.path.exists(path):
            os.mkdir(path)
    
    if not os.path.exists(parent):
        rmkdir(parent)
    if not os.path.exists(path):
        os.mkdir(path)
        
def remove_vertices_and_faces(mesh, vertices_to_remove):
    """
    从网格中移除指定的顶点及其相关的面片。
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 输入的三角形网格。
        vertices_to_remove (list): 要删除的顶点索引列表。
        
    Returns:
        o3d.geometry.TriangleMesh: 移除指定顶点和面片后的网格。
    """
    # 获取网格的顶点和面片
    vertices = np.asarray(mesh.vertices)
    if len(mesh.vertex_colors) > 0:
        vertices_colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)

    # 标记需要删除的顶点
    vertices_mask = np.ones(len(vertices), dtype=bool)
    vertices_mask[vertices_to_remove] = False

    # 过滤掉包含这些顶点的三角形
    triangles_mask = np.all(vertices_mask[triangles], axis=1)

    # 更新顶点和面片列表
    new_vertices = vertices[vertices_mask]
    new_triangles = triangles[triangles_mask]

    # 映射新的顶点索引
    old_to_new_indices = np.cumsum(vertices_mask) - 1
    new_triangles = old_to_new_indices[new_triangles]

    # 创建新的网格
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    if len(mesh.vertex_colors) > 0:
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(vertices_colors[vertices_mask])
    return new_mesh

def clean_mesh2(verts,faces,xyz,k=30):
    # 由点建立kd树
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    o3d_mesh = clean_mesh(o3d_mesh,pc,k)
    verts = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return verts,faces
    


def clean_mesh(mesh,pc,k=30):
    # 由点建立kd树
    points_onmesh = np.asarray(mesh.vertices)
    kdtree = cKDTree(points_onmesh)
    visited = np.zeros(len(points_onmesh))
    xyz = np.asarray(pc.points)
    _, idx = kdtree.query(xyz, k)
    idx = idx.flatten()
    visited[idx] = 1
    mesh = remove_vertices_and_faces(mesh,np.where(visited==0)[0])
    return mesh
    

def poission_rec(input_file_name,output_file_name,inv=True,clean_k=30,btype=3,depth=12,colors=False):
    poission_rec_exe_path = "D:\Documents\zhudoongli\CG\project\PoissonRecon\Bin/x64\Release/PoissonRecon.exe"
    cmd = poission_rec_exe_path + " --in " + input_file_name + " --out " + output_file_name  + " --depth %d"%depth + " --btype " + str(btype) 
    if colors:
        cmd += " --colors"
    os.system(cmd)
    mesh = o3d.io.read_triangle_mesh(output_file_name)
    pcd = o3d.io.read_point_cloud(input_file_name)
    if clean_k>0:
        mesh = clean_mesh(mesh,pcd,clean_k)
    if inv:
        # 翻转面片
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:,::-1])
    o3d.io.write_triangle_mesh(output_file_name.replace(".ply","_clean.ply"),mesh)
    return


def extract_surface_from_scalar_field(scalar_field, level, 
                                      resolution,bbox=np.array([[-1,1],[-1,1],[-1,1]]),mask=None,
                                      save_path=None):
    """
    从3D标量场中提取等值面
    参数:
        scalar_field: N*N*N的numpy数组，表示标量场
        level: float，等值面的值（默认0.5）
        spacing: tuple，体素在x,y,z方向上的间距（默认都是1.0）
    返回:
        verts: 顶点坐标数组 (n_verts, 3)
        faces: 面片索引数组 (n_faces, 3)
    """
    # spacing = (2/(resolution-1),2/(resolution-1),2/(resolution-1))
    spacing = np.array([bbox[0,1]-bbox[0,0],bbox[1,1]-bbox[1,0],bbox[2,1]-bbox[2,0]])/(resolution-1)
    
    try:
        # 对mask进行腐蚀
        if mask is not None:
            import skimage.morphology as morphology
            kernel = morphology.ball(1)
            mask = morphology.binary_erosion(mask,kernel)

        # 使用marching cubes提取等值面
        verts, faces, normals, values = measure.marching_cubes(
            scalar_field,
            level=level,
            spacing=spacing,
            allow_degenerate=False,
            mask=mask
        )
        verts = verts + [bbox[0,0],bbox[1,0],bbox[2,0]]
        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),o3d.utility.Vector3iVector(faces)))
        return verts, faces
        
    except Exception as e:
        print(f"Error in marching cubes: {str(e)}")
        return None, None

def create_uniform_grid(resolution, bbox=np.array([[-1, 1], [-1, 1], [-1, 1]])):
    """
    创建均匀采样的 3D 网格采样点
    @return points: (N, 3) array - 网格采样点
    @return grid_shape: (3,) tuple - 网格形状
    """
    numpy_dtype = get_numpy_dtype()
    bbox = bbox.astype(numpy_dtype)
    
    x = np.linspace(bbox[0, 0], bbox[0, 1], resolution, dtype=numpy_dtype)
    y = np.linspace(bbox[1, 0], bbox[1, 1], resolution, dtype=numpy_dtype)
    z = np.linspace(bbox[2, 0], bbox[2, 1], resolution, dtype=numpy_dtype)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(numpy_dtype)
    grid_shape = (resolution, resolution, resolution)
    return points, grid_shape

# def allocate_points_to_grid(points,grid):
#     '''
#     将点云分配到网格中
#     @param points: (N, 3) array - 点云
#     @param grid: (M, 3) array - 网格位置
#     @return : (N,) array - 点云在网格中的索引
#     '''
#     kdtree = cKDTree(grid)
#     _,idx = kdtree.query(points,k=1)
#     return idx

'''
创建mask
tomask: 需要mask的网格坐标
source_points: 需要mask的点云坐标
k: 在source_points附近的k个点会被mask
'''
def create_mask_by_k(tomask,source_points,k):
    mask = np.zeros(tomask.shape[0],dtype=np.bool_)
    kdtree = cKDTree(tomask)
    _,idx = kdtree.query(source_points,k=k,workers=-1)
    idx = idx.flatten()
    mask[idx] = True
    return mask
    
def clean_bad_data(points,normals):
    mask = np.linalg.norm(normals,axis=1) > 0.01
    print("There is %d points removed" % (len(points) - np.sum(mask)))
    points = points[mask]
    normals = normals[mask]
    return points,normals

def fun0(G):
    G[G!=0] = 1/G[G!=0]
    return G

def fun1(G):
    G[G!=0] = 1/G[G!=0]
    G = np.clip(G,0,15)
    G = np.exp(G)
    return G

def Gaussian(G,sigma=1):
    zero_mask = G!=0
    # 计算高斯权重
    G[zero_mask] = np.exp(-G[zero_mask]**2/(2*sigma**2))
    G[zero_mask] += 1e-10
    return G
    

def get_kernel_correspond_to_connectivity(connectivity):
    kernel = np.zeros((3,3,3),dtype=np.uint8)
    if connectivity >= 6:
        kernel[1,1,:] = 1
        kernel[:,1,1] = 1
        kernel[1,:,1] = 1
    if connectivity >= 18:
        kernel[1,:,:] = 1
        kernel[:,1,:] = 1
        kernel[:,:,1] = 1
    if connectivity >= 26:
        kernel[0,:,:] = 1
        kernel[:,0,:] = 1
        kernel[:,:,0] = 1
    return kernel


def voxelize_points(points,voxel_size,volume_resolution,if_dilate=False):
    cube_dilate = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, -1],
                [0, -1, 0],
                [0, 1, 1],
                [0, -1, 1],
                [0, 1, -1],
                [0, -1, -1],

                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, -1],
                [1, -1, 0],
                [1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                [1, -1, -1],

                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 1, 0],
                [-1, 0, -1],
                [-1, -1, 0],
                [-1, 1, 1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
            ]
        ) / (volume_resolution * 4 - 1)
    
    if if_dilate:
        points = points[np.newaxis] + cube_dilate[...,np.newaxis,:]
        points = points.reshape(-1,3)
        

'''
体素网格计算梯度
'''
import torch
def compute_gradient(values:torch.Tensor, mask:torch.Tensor):
    """
    计算梯度
    """
    masked_values = values.clone()
    masked_values[~mask] = np.nan
    dx = torch.diff(masked_values, dim=0)
    dx = torch.cat([dx,torch.zeros_like(dx[-1,:]).unsqueeze(0)],dim=0)
    dy = torch.diff(masked_values, dim=1)
    dy = torch.cat([dy,torch.zeros_like(dy[:,-1]).unsqueeze(1)],dim=1)
    dz = torch.diff(masked_values, dim=2)
    dz = torch.cat([dz,torch.zeros_like(dz[:,:,-1]).unsqueeze(2)],dim=2)
    grad_length = torch.sqrt(dx**2 + dy**2 + dz**2)
    grad_length[torch.isnan(grad_length)] = 0
    return grad_length

def cal_normal_acc(gt_normal:np.array, pred_normal:np.array) -> float:
    """
    gt_normal: (N, 3) array - ground truth 法向量
    pred_normal: (N, 3) array - 预测法向量
    """
    dot_product = np.sum(gt_normal * pred_normal, axis=1)
    if_inv = dot_product < 0
    iif_inv = dot_product > 0
    acc = max(np.mean(if_inv), np.mean(iif_inv))
    return acc