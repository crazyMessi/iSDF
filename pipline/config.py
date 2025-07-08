import torch

basic_config = {
    # Data parameters
    "num_samples": 2048,  # Number of points to sample from each mesh
    # Model general parameters
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "R": 256,  # Resolution for GT SDF grid and final predicted SDF grid (R x R x R)
    "r": 256,  # Resolution for intermediate voxel grid from predicted normals (r x r x r)

    # PointNet (f1) parameters
    "pointnet_input_channels": 3, # x,y,z for points
    "pointnet_output_dim": 256,

    # PCA normal estimation parameters
    "pca_knn": 10, # k-nearest neighbors for PCA normal estimation

    # Training parameters
    "batch_size": 1,
    "epochs": 100,
    "learning_rate": 1e-5,
    "log_interval": 1, # Print log every N batches
    "save_checkpoint_interval": 10, # Save checkpoint every N epochs

    # Other
    "use_torch_dataloader": True, # From original train_per_pc.py
    "loss_fn": "CustomLoss",
} 

def get_config(*args, **kwargs):
    config = basic_config.copy()
    if kwargs.get("dataset_name") != None:
        if kwargs.get("dataset_name") == "pcpnet":
            config["data_path"] = f"/mnt/lizd/workdata/iSDF/pcpnet/poission_mesh/"
            config["save_path"] = f"/mnt/lizd/workdata/iSDF/pcpnet/poission_mesh/temp/"
        else:
            import os
            if os.path.exists(f"/mnt/lizd/workdata/iSDF/{kwargs.get('dataset_name')}"):
                config["data_path"] = f"/mnt/lizd/workdata/iSDF/{kwargs.get('dataset_name')}/"
                config["save_path"] = f"/mnt/lizd/workdata/iSDF/{kwargs.get('dataset_name')}/temp/"
            else:
                raise ValueError(f"Dataset name {kwargs.get('dataset_name')} not supported")
    else:
        config["data_path"] = "/mnt/lizd/workdata/iSDF/mesh_segment/"
        config["save_path"] = "/mnt/lizd/workdata/iSDF/mesh_segment/temp/"
        
    
    return config