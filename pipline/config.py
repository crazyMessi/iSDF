import torch

config = {
    # Data parameters
    "data_path": "/mnt/lizd/workdata/iSDF/mesh_segment/",
    "save_path": "/mnt/lizd/workdata/iSDF/mesh_segment/temp/", # For dataloader caching or intermediate results
    "num_samples": 2048,  # Number of points to sample from each mesh

    # Model general parameters
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "R": 64,  # Resolution for GT SDF grid and final predicted SDF grid (R x R x R)
    "r": 64,  # Resolution for intermediate voxel grid from predicted normals (r x r x r)

    # PointNet (f1) parameters
    "pointnet_input_channels": 3, # x,y,z for points
    "pointnet_output_dim": 256,

    # PCA normal estimation parameters
    "pca_knn": 16, # k-nearest neighbors for PCA normal estimation

    # Sparse Convolutional Encoder (f2) parameters
    "sparse_cnn_input_channels": 1, # Assuming the voxel grid from normals has 1 feature channel
    "sparse_cnn_output_dim": 256,

    # Decoder parameters
    "decoder_input_dim": 512, # pointnet_output_dim + sparse_cnn_output_dim
    # Decoder will output R x R x R grid

    # Training parameters
    "batch_size": 1,
    "epochs": 100,
    "learning_rate": 1e-4,
    "log_interval": 1, # Print log every N batches

    # Other
    "use_torch_dataloader": True, # From original train_per_pc.py
} 