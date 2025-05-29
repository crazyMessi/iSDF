import numpy as np
from scipy.spatial import KDTree

def dir_update_normal(points,mesh_vertices,mesh_faces,k=10):
    '''
    用距离近的面法向量更新当前点法向量
    '''

    
    # 计算每个面片的中心点和法向量
    v0 = mesh_vertices[mesh_faces[:, 0]]  # (F, 3)
    v1 = mesh_vertices[mesh_faces[:, 1]]
    v2 = mesh_vertices[mesh_faces[:, 2]]
    
    # 计算面片中心
    face_centers = (v0 + v1 + v2) / 3.0  # (F, 3)
    
    # 计算面片法向量
    face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3)
    face_normal_norms = np.linalg.norm(face_normals, axis=1, keepdims=False)
    face_normals[face_normal_norms < 1e-8] = np.random.randn(np.sum(face_normal_norms < 1e-8), 3)
    face_normal_norms[face_normal_norms < 1e-8] = np.linalg.norm(face_normals[face_normal_norms < 1e-8], axis=1, keepdims=False)
    if np.sum(face_normal_norms < 1e-8) > 0:
        print(f"There is {np.sum(face_normal_norms < 1e-8)} face normals are too small")
    face_normals = face_normals / face_normal_norms[:,None]
        
    # 计算每个点最近的k个面片
    tree = KDTree(face_centers)
    _, indices = tree.query(points, k=k)  # (N, k)
    
    # 计算每个点最近的k个面片的法向量
    point_normals = np.zeros_like(points, dtype=np.float32)  # (N, 3)
    
    point_normals = face_normals[indices].sum(axis=1)
    
    
    
    # 归一化
    norms = np.linalg.norm(point_normals, axis=1, keepdims=True)
    if len(norms[norms < 1e-8]) > 0:
        print(f"Warning: {len(norms[norms < 1e-8])} normals are too small")
    norms[norms < 1e-8] = 1.0
    point_normals = point_normals / norms
    
    return point_normals
    


def compute_point_normals_from_mesh(points: np.ndarray, 
                                  mesh_vertices: np.ndarray, 
                                  mesh_faces: np.ndarray,
                                  k: int = 10) -> np.ndarray:
    """
    计算点云的法向量
    
    方法:
    1. 计算每个面片的法向量
    2. 对每个面片找到k个最近的点
    3. 将面片法向量累加到这些点上
    4. 归一化得到最终法向量
    
    Args:
        points: (N, 3) array - 需要计算法向量的点云
        mesh_vertices: (M, 3) array - 输入三角网格的顶点
        mesh_faces: (F, 3) array - 输入三角网格的面片
        k: int - 每个面片影响的近邻点数量
        
    Returns:
        (N, 3) array - 点云的法向量
    """
    # 1. 建立点云的KD树
    tree = KDTree(points)
    
    # 2. 计算每个面片的中心点和法向量
    v0 = mesh_vertices[mesh_faces[:, 0]]  # (F, 3)
    v1 = mesh_vertices[mesh_faces[:, 1]]
    v2 = mesh_vertices[mesh_faces[:, 2]]
    
    # 计算面片中心
    face_centers = (v0 + v1 + v2) / 3.0  # (F, 3)
    
    # 计算面片法向量
    face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3)
    face_normal_norms = np.linalg.norm(face_normals, axis=1, keepdims=False)
    face_normals[face_normal_norms < 1e-8] = np.random.randn(np.sum(face_normal_norms < 1e-8), 3)
    face_normal_norms[face_normal_norms < 1e-8] = np.linalg.norm(face_normals[face_normal_norms < 1e-8], axis=1, keepdims=False)
    print(f"There is {np.sum(face_normal_norms < 1e-8)} face normals are too small")
    face_normals = face_normals / face_normal_norms[:,None]
    
    
    # 3. 初始化输出法向量数组
    point_normals = np.zeros_like(points, dtype=np.float32)  # (N, 3)
    
    # 4. 批量查询每个面片的k个最近点
    _, indices = tree.query(face_centers, k=k)  # (F, k)
    
    # 5. 将面片法向量累加到近邻点
    for i in range(k):
        point_normals[indices[:,i]] += face_normals
    
    # 6. 归一化得到最终法向量
    norms = np.linalg.norm(point_normals, axis=1, keepdims=True)
    if len(norms[norms < 1e-8]) > 0:
        print(f"Warning: {len(norms[norms < 1e-8])} normals are too small")
    norms[norms < 1e-8] = 1.0
    point_normals = point_normals / norms
    assert np.isnan(point_normals).sum() == 0, "法向量中存在nan"
    assert np.isinf(point_normals).sum() == 0, "法向量中存在inf"
    return point_normals





