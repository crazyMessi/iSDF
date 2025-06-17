import torch
import numpy as np
import open3d as o3d
import trimesh
import os
import time
import argparse


def normalize_mesh(mesh_path):
    scene = trimesh.load(mesh_path, process=False, force='scene')
    meshes = []
    for node_name in scene.graph.nodes_geometry:
        geom_name = scene.graph[node_name][1]
        geometry = scene.geometry[geom_name]
        transform = scene.graph[node_name][0]
        if isinstance(geometry, trimesh.Trimesh):
            geometry.apply_transform(transform)
            meshes.append(geometry)

    mesh = trimesh.util.concatenate(meshes)

    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = max(mesh.bounding_box.extents)
    mesh.apply_scale(2.0 / scale * 0.5)

    return mesh

def load_quantized_mesh_original(
    mesh_path, 
    volume_resolution=256,
    use_normals=True,
    pc_sample_number=4096000,
):
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
        
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    faces = np.asarray(mesh.triangles)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    voxelization_mesh = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=1. / volume_resolution,
            min_bound=[-0.5, -0.5, -0.5],
            max_bound=[0.5, 0.5, 0.5]
        )
    voxel_mesh = np.asarray([voxel.grid_index for voxel in voxelization_mesh.get_voxels()])

    points_normals_sample = trimesh.Trimesh(vertices=vertices, faces=faces).sample(count=pc_sample_number, return_index=True)
    points_sample = points_normals_sample[0].astype(np.float32)
    voxelization_points = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(
                    np.clip(
                        (points_sample[np.newaxis] + cube_dilate[..., np.newaxis, :]).reshape(-1, 3),
                        -0.5 + 1e-6, 0.5 - 1e-6)
                    )
                ),
            voxel_size=1. / volume_resolution,
            min_bound=[-0.5, -0.5, -0.5],
            max_bound=[0.5, 0.5, 0.5]
        )
    voxel_points = np.asarray([voxel.grid_index for voxel in voxelization_points.get_voxels()])
    voxels = torch.Tensor(np.unique(np.concatenate([voxel_mesh, voxel_points]), axis=0))

    if use_normals:
        mesh.compute_triangle_normals()
        normals_sample = np.asarray(
                            mesh.triangle_normals
                        )[points_normals_sample[1]].astype(np.float32)
        points_sample = torch.cat((torch.Tensor(points_sample), torch.Tensor(normals_sample)), axis=-1)
    
    return voxels, points_sample


# def voxelize_pointcloud(pointcloud, volume_resolution):
    












default_mesh_path = "/mnt/lizd/workdata/iSDF/mesh_segment/SceneNN_054max_part_tri_1.ply"
if __name__ == "__main__":
    # Usage: `python inference.py --mesh-path "assets/examples/loong.obj" --output-dir "outputs/" --config "configs/triposfVAE_1024.yaml"`
    parser = argparse.ArgumentParser("TripoSF Reconstruction")
    parser.add_argument("--output-dir", default="outputs/", help="path to output folder")
    parser.add_argument("--mesh-path", type=str, default=default_mesh_path, help="the input mesh to be reconstructed")
    parser.add_argument("--volume-resolution", type=int, default=256, help="the resolution of the voxel grid")
    parser.add_argument("--pc-sample-number", type=int, default=409600, help="the number of points to sample from the mesh")
    
    args, extras = parser.parse_known_args()
    save_name = os.path.split(args.mesh_path)[-1].split(".")[0]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Mesh Normalizing...")
    preprocess_start = time.time()
    mesh_gt = normalize_mesh(args.mesh_path)
    save_path_gt = f"{args.output_dir}/{save_name}_gt.ply"
    trimesh.Trimesh(vertices=mesh_gt.vertices.tolist(), faces=mesh_gt.faces.tolist()).export(save_path_gt)
    preprocess_end = time.time()
    print(f"Mesh Normalizing Time: {(preprocess_end - preprocess_start):.2f}")
    
    print(f"Mesh Loading...")
    input_loading_start = time.time()
    sparse_voxels, points_sample = load_quantized_mesh_original(
                                                            save_path_gt, 
                                                            volume_resolution=args.volume_resolution, 
                                                            use_normals=False, 
                                                            pc_sample_number=args.pc_sample_number
                                                        )
    input_loading_end = time.time()
    print(f"Mesh Loading Time: {(input_loading_end - input_loading_start):.2f}")
    # save points_sample
    save_path_points = f"{args.output_dir}/{save_name}_points.ply"
    o3d.io.write_point_cloud(save_path_points, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sample)))
    # save sparse_voxels
    save_path_voxels = f"{args.output_dir}/{save_name}_voxels.ply"
    o3d.io.write_point_cloud(save_path_voxels, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sparse_voxels)))
   