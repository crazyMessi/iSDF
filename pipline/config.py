import torch
import json
basic_config = {
    # Data parameters
    "num_samples": 10000,  # Number of points to sample from each mesh
    # Model general parameters
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "R": 128,  # Resolution for GT SDF grid and final predicted SDF grid (R x R x R)
    "r": 128,  # Resolution for intermediate voxel grid from predicted normals (r x r x r)

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
    "k_for_mask": 216, # 在sample point附近的1920个点会被mask
} 
local_config = json.load(open("config/local_config.json"))

def get_config(*args, **kwargs):
    # database_path = "/mnt/lizd/workdata/iSDF/"
    database_path = local_config.get("database_path")
    config = basic_config.copy()
    config["experiment_output_path"] = local_config.get("experiment_output_path")
    if kwargs.get("dataset_name") != None:
        if kwargs.get("dataset_name") == "pcpnet":
            config["data_path"] = f"{database_path}/pcpnet/poission_mesh/"
            config["save_path"] = f"{database_path}/pcpnet/poission_mesh/temp/"
        else:
            import os
            if os.path.exists(f"{database_path}/{kwargs.get('dataset_name')}"):
                config["data_path"] = f"{database_path}/{kwargs.get('dataset_name')}/"
                config["save_path"] = f"{database_path}/{kwargs.get('dataset_name')}/temp/"
            else:
                raise ValueError(f"Dataset name {kwargs.get('dataset_name')} not supported")
    else:
        config["data_path"] = f"{database_path}/mesh_segment/"
        config["save_path"] = f"{database_path}/mesh_segment/temp/"
        
    
    return config