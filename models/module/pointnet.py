import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from torch_scatter import scatter_mean

class TNet(nn.Module):
    """变换网络，用于学习输入变换矩阵"""
    
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        batchsize = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # 初始化为单位矩阵
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x


class PointNetEncoder(nn.Module):
    """PointNet编码器，将点云编码为固定长度的特征向量"""
    
    def __init__(self, input_dim=3, output_dim=1024, use_tnet=True):
        """
        Args:
            input_dim: 输入点云的维度（例如3表示xyz坐标）
            output_dim: 输出特征向量的维度
            use_tnet: 是否使用变换网络
        """
        super(PointNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_tnet = use_tnet
        
        # 输入变换网络
        if self.use_tnet:
            self.input_transform = TNet(k=input_dim)
            self.feature_transform = TNet(k=64)
        
        # 点级特征提取网络
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # 全局特征处理
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Args:
            x: 输入点云，形状为 (batch_size, input_dim, num_points)
        
        Returns:
            编码后的特征向量，形状为 (batch_size, output_dim)
        """
        batch_size, _, num_points = x.size()
        
        # 输入变换
        if self.use_tnet:
            trans_input = self.input_transform(x)
            x = x.transpose(2, 1)  # (B, N, input_dim)
            x = torch.bmm(x, trans_input)  # 应用变换矩阵
            x = x.transpose(2, 1)  # (B, input_dim, N)
        
        # 第一层特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 特征变换
        if self.use_tnet:
            trans_feat = self.feature_transform(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        
        # 继续特征提取
        pointfeat = x  # 保存点级特征用于分割任务
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # 全局最大池化
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # 全连接层生成最终特征
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_regularization_loss(self):
        """获取特征变换的正则化损失"""
        if not self.use_tnet:
            return 0
        
        # 这个方法需要在forward过程中调用才能获取变换矩阵
        # 通常在训练循环中使用
        return 0

def scale_tensor(
    dat, inp_scale=None, tgt_scale=None
):
    """
    Scales tensor values from input range to target range.
    
    Args:
        dat: Input tensor to scale
        inp_scale: Input range tuple (min, max), default is (-0.5, 0.5)
        tgt_scale: Target range tuple (min, max), default is (0, 1)
    
    Returns:
        Scaled and clamped tensor
    """
    if inp_scale is None:
        inp_scale = (-0.5, 0.5)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    assert tgt_scale[1] > tgt_scale[0] and inp_scale[1] > inp_scale[0]
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat.clamp(tgt_scale[0] + 1e-6, tgt_scale[1] - 1e-6)

# Resnet Blocks for pointnet
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        """
        Initialize ResNet block with fully connected layers.
        
        Args:
            size_in: Input feature dimension
            size_out: Output feature dimension (defaults to size_in if None)
            size_h: Hidden dimension size (defaults to min(size_in, size_out) if None)
        """
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.GELU(approximate="tanh")

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.xavier_uniform_(self.fc_0.weight)
        if self.fc_0.bias is not None:
            nn.init.constant_(self.fc_0.bias, 0)
        if self.shortcut is not None:
            nn.init.xavier_uniform_(self.shortcut.weight)
            if self.shortcut.bias is not None:
                nn.init.constant_(self.shortcut.bias, 0)
        
        nn.init.xavier_uniform_(self.fc_1.weight)
        if self.fc_1.bias is not None:
            nn.init.constant_(self.fc_1.bias, 0)
        
        
    def forward(self, x):
        """
        Forward pass of ResNet block.
        
        Args:
            x: Input tensor
        
        Returns:
            Residual connection output (input + processed features)
        """
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class LocalPoolPointnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, hidden_dim=128, scatter_type='mean', n_blocks=5):
        """
        Initialize LocalPoolPointnet module for point cloud processing.
        forward:
            输入[B,N,3]的点云坐标,输出[B*N,out_channels]的体素特征
        
        Args:
            in_channels: Input channel dimension (default: 3 for xyz coordinates)
            out_channels: Output feature dimension
            hidden_dim: Hidden layer dimension
            scatter_type: Type of scatter operation ('mean' supported)
            n_blocks: Number of ResNet blocks to use
        """
        super().__init__()
        self.scatter_type = scatter_type
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.fc_pos = nn.Linear(in_channels, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, out_channels)

        if self.scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('Incorrect scatter type')
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize network weights with Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.fc_pos.weight)
        if self.fc_pos.bias is not None:
            nn.init.constant_(self.fc_pos.bias, 0)

        nn.init.xavier_uniform_(self.fc_c.weight)
        if self.fc_c.bias is not None:
            nn.init.constant_(self.fc_c.bias, 0)

    def convert_to_sparse_feats(self, c, sparse_coords):
        """
        Convert dense grid features to sparse features based on sparse coordinates.
        
        Args:
            sparse_coords: Tensor [Nx, 4], point to sparse indices (batch_idx, x, y, z)
            c: Tensor [B, res, C], input features of each grid
        
        Returns:
            feats_new: Tensor [Nx, C], aggregated grid features for each sparse point
        """
        feats_new = torch.zeros((sparse_coords.shape[0], c.shape[-1]), device=c.device, dtype=c.dtype)
        offsets = 0
        
        batch_nums = copy.deepcopy(sparse_coords[..., 0])
        for i in range(len(c)):
            coords_num_i = (batch_nums == i).sum()
            feats_new[offsets: offsets + coords_num_i] = c[i, : coords_num_i]
            offsets += coords_num_i
        return feats_new

    def generate_sparse_grid_features(self, index, c, max_coord_num):
        """
        Generate sparse grid features by scattering point features.
        
        Args:
            index: Tensor [B, 1, Np], sparse indices of each point
            c: Tensor [B, Np, C], input features of each point
            max_coord_num: Maximum number of coordinates per batch
        
        Returns:
            c_out: Tensor [B, res, C], scattered features in grid
        """
        # scatter grid features from points
        bs, fea_dim = c.size(0), c.size(2)
        res = max_coord_num
        c_out = c.new_zeros(bs, self.out_channels, res)
        # Permute dimensions for scatter operation:
        # From [B, Np, C] to [B, C, Np], scatter along dimension 2, then permute back
        c_out = scatter_mean(c.permute(0, 2, 1), index, out=c_out).permute(0, 2, 1) # B x res X C
        return c_out

    def pool_sparse_local(self, index, c, max_coord_num):
        """
        Pool features locally for sparse point representation.
        
        Args:
            index: Tensor [B, 1, Np], sparse indices of each point
            c: Tensor [B, Np, C], input features of each point
            max_coord_num: Maximum number of coordinates per batch
        
        Returns:
            Tensor [B, Np, C], aggregated grid features for each point
        """
        bs, fea_dim = c.size(0), c.size(2)
        res = max_coord_num
        c_out = c.new_zeros(bs, fea_dim, res)
        
        # Scatter features to grid positions using mean operation
        # Permute to [B, C, Np] for scatter operation
        c_out = self.scatter(c.permute(0, 2, 1), index, out=c_out)

        # Gather feature back to points by expanding index to match feature dimension
        # and gather along the spatial dimension
        c_out = c_out.gather(dim=2, index=index.expand(-1, fea_dim, -1))
        return c_out.permute(0, 2, 1)

    @torch.no_grad()
    def coordinate2sparseindex(self, x, sparse_coords, res):
        """
        Convert point coordinates to sparse indices.
        
        Args:
            x: Tensor [B, Np, 3], points scaled at ([0, 1] * res)
            sparse_coords: Tensor [Nx, 4] ([batch_number, x, y, z])
            res: Int, resolution of the grid index
        
        Returns:
            sparse_index: Tensor [B, 1, Np], sparse indices of each point
            
        Note:
            This converts 3D coordinates to flattened 1D indices using
            index = (x*res + y)*res + z, then finds the position of each
            point's index in the sorted sparse indices array using searchsorted.
        """
        B = x.shape[0]
        sparse_index = torch.zeros((B, x.shape[1]), device=x.device, dtype=torch.int64)
        
        # Flatten 3D coordinates to 1D index using (x*res + y)*res + z
        index = (x[..., 0] * res + x[..., 1]) * res + x[..., 2]
        sparse_indices = copy.deepcopy(sparse_coords)
        sparse_indices[..., 1] = (sparse_indices[..., 1] * res + sparse_indices[..., 2]) * res + sparse_indices[..., 3]
        sparse_indices = sparse_indices[..., :2]
        
        for i in range(B):
            mask_i = sparse_indices[..., 0] == i
            coords_i = sparse_indices[mask_i, 1]
            coords_num_i = len(coords_i)
            # Find position of each point's index in the sorted sparse indices array
            sparse_index[i] = torch.searchsorted(coords_i, index[i])
                
        return sparse_index[:, None, :]

    def forward(self, p, sparse_coords, res=64, bbox_size=(-0.5, 0.5)):
        """
        Forward pass of LocalPoolPointnet.
        
        Args:
            p: Tensor [B, Np, D], point cloud coordinates (and normals if D=6)
            sparse_coords: Tensor [Nx, 4] ([batch_number, x, y, z]), sparse coordinates Nx = B*Np
            res: Resolution of the grid index
            bbox_size: Bounding box size for normalization
        
        Returns:
            sparse_pc_feats: [Nx, out_channels], features for each sparse point
            
            tips:可以使用sp.SparseTensor(sparse_coords, sparse_pc_feats)来存储稀疏特征
        
        Process:
            1. Scale points to grid coordinates and compute offsets to grid centers
            2. Convert coordinates to sparse indices
            3. Process point features through ResNet blocks
            4. Pool features locally for each iteration
            5. Generate final sparse grid features
        """
        batch_size, T, D = p.size()
        max_coord_num = 0
        for i in range(batch_size):
            max_coord_num = max(max_coord_num, (sparse_coords[..., 0] == i).sum().item() + 5)
        
        if D == 6:
            p, normals = p[..., :3], p[..., 3:]

        # Scale coordinates to [0, res] range, then compute offset to grid center (-1 to 1)
        coord = (scale_tensor(p, inp_scale=bbox_size) * res)
        p = 2 * (coord - (coord.floor() + 0.5)) # dist to the centroids, [-1., 1.]
        index = self.coordinate2sparseindex(coord.long(), sparse_coords, res)

        if D == 6:
            p = torch.cat((p, normals), dim=-1)
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            # Pool features locally and concatenate with current features
            # 先对每个voxel内的点进行mean得到voxel的特征，然后将voxel逆向映射到点云上
            pooled = self.pool_sparse_local(index, net, max_coord_num=max_coord_num)
            
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        feats = self.generate_sparse_grid_features(index, c, max_coord_num=max_coord_num)
        feats = self.convert_to_sparse_feats(feats, sparse_coords)
        torch.cuda.empty_cache()
        return feats
