"""
Mesh Partition Module

这个包提供用于三角网格分区的工具和函数
"""

try:
    # 首先尝试导入C++扩展模块
    from .mesh_partition import process_single_mesh
    
    # 然后导入Python接口
    from .mesh_partition import partition_mesh, visualize_partition
except ImportError:
    # 如果C++扩展模块未编译，给出有用的错误信息
    import warnings
    warnings.warn(
        "无法导入C++模块'mesh_partition'。请确保已经正确编译C++扩展。\n"
        "可以通过在包目录中运行'pip install -e .'来编译该模块。"
    )

__version__ = '0.0.1' 