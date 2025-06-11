# Train Per Point Cloud with PartSeg Dataset

This script (`train_per_pc.py`) combines the network architecture from `train_mix.py` with the dataset splitting methodology from `train_partseg.py`. It performs training on each point cloud by:

1. **Feature Encoding**: Each point cloud is encoded into feature vectors using ShapeFeatureExtractor
2. **WNF Field Computation**: For each point cloud, a Weighted Normal Field (WNF) is computed and encoded
3. **Classification**: The combined features are used for point cloud classification

## Key Features

- Uses ShapeNet Part Segmentation dataset with trainval/test splits
- Encodes both shape features and WNF features for each point cloud
- Combines features for classification tasks
- Supports normal channels and data augmentation

## Usage

### Basic Training
```bash
cd pipline
python train_per_pc.py --normal --batch_size 8 --epoch 100
```

### With Custom Configuration
```bash
python train_per_pc.py --config_dir ../config --normal --batch_size 16 --learning_rate 0.0005
```

### Command Line Arguments

- `--config_dir`: Directory containing configuration files (default: '../config')
- `--batch_size`: Batch size during training (default: 16)
- `--epoch`: Number of epochs (default: 251)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--gpu`: GPU device to use (default: '0')
- `--optimizer`: Optimizer type - 'Adam' or 'SGD' (default: 'Adam')
- `--log_dir`: Custom log directory name
- `--npoint`: Number of points per point cloud (default: 2048)
- `--normal`: Use normal channels (recommended)
- `--data_root`: Path to ShapeNet dataset

## Network Architecture

The training pipeline consists of:

1. **ShapeFeatureExtractor**: Encodes point clouds (with normals) into shape feature vectors
2. **SparseVoxelEncoder**: Encodes WNF fields into feature vectors
3. **Classifier**: Multi-layer perceptron for classification

## Output

The script saves:
- Training logs in `./log/per_pc_training/[timestamp]/logs/`
- Model checkpoints in `./log/per_pc_training/[timestamp]/checkpoints/`
- Best model based on test accuracy

## Configuration

The script uses configuration files from the `config` directory:
- `base_config.json`: Base configuration
- `train_config.json`: Training-specific configuration

Key configuration parameters:
- Model dimensions (shape_encoder, wnf_encoder output dimensions)
- Field resolution and bounding box
- Device and data type settings

## Requirements

- PyTorch
- NumPy
- Open3D
- SciPy
- spconv (for sparse convolutions)
- tqdm

## Data Format

Expects ShapeNet Part Segmentation dataset with:
- Point coordinates (x, y, z)
- Optional normal vectors (nx, ny, nz)
- Category labels
- Part segmentation labels

## Notes

- WNF computation is computationally intensive - consider reducing batch size if memory issues occur
- The script includes fallback mechanisms for WNF computation failures
- Normal estimation is performed automatically if normals are not provided
- Training includes data augmentation (scaling, shifting) similar to the original PartSeg training 