### 环境配置
在anaconda中安装cuda12.4 然后安装pytorch2.4
注意 需要配置PATH使得cuda12.4在/usr/local/cuda-12.4/bin之前

安装trellis的依赖

安装pysdf
```
pip install pysdf
```
如果报错缺失头文件crypt，则安装crypt
```
conda install --channel=conda-forge libxcrypt
```

安装flash_attn==2.7.4.post1
```
pip install flash_attn==2.7.4.post1
```

如果flash_attn还是有问题，请调整cuda、pytorch、numpy版本



