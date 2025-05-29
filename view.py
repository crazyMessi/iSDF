import numpy as np
import polyscope as ps
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

def read_grid_file(filename):
    with open(filename, 'rb') as f:
        # 读取分辨率（一个 int 值）
        res = np.fromfile(f, dtype=np.int32, count=1)[0]

        # 读取浮点数数据
        total_floats = res * res * res
        grid_data = np.fromfile(f, dtype=np.float32, count=total_floats)

        # 将数据重塑为三维数组
        grid_data = grid_data.reshape((res, res, res))

    return grid_data


def view_data(labels_out,bound_low=[0,0,0],bound_high = [1,1,1]):
    ps.init()
    # define the resolution and bounds of the grid

    # color = cm.get_cmap("Paired")(labels_out).astype(float)[...,:3].reshape((-1,3))
    # register the grid
    ps_grid = ps.register_volume_grid("sample grid", labels_out.shape, bound_low, bound_high)


    # add a scalar function on the grid
    ps_grid.add_scalar_quantity("node scalar1", labels_out,
                                defined_on='nodes', enabled=True)

    ps.show()


# filename = "D:\WorkData\ipsr_explore\out\it_1_dp_9_nb_10_sd_10_pt_10.000000\scan600/temp/-1.grid"
# data = read_grid_file(filename)

# view_data(data)