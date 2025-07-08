import os
os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lzd_models.slat_net import test_SparseVoxelDownsampler
test_SparseVoxelDownsampler()