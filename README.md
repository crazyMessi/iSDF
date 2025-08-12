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


# Git文件备份工具

这个工具可以将当前被git追踪的文件复制到指定位置，支持文件大小统计、用户确认等功能。

## 功能特点

1. **只复制git追踪的文件**: 使用`git ls-files`和`git check-ignore`确保只复制真正被git追踪的文件
2. **文件大小统计**: 在复制前统计所有文件的总大小
3. **用户确认**: 当文件大小超过设定限制时，会询问用户是否继续
4. **灵活的排除模式**: 可以排除特定类型的文件
5. **目录结构保持**: 可选择是否保持原始目录结构
6. **时间戳目录**: 可选择是否创建带时间戳的备份目录

## 使用方法

### 1. 命令行使用

```bash
# 基本用法
python git_backup.py ./backups

# 排除某些文件类型
python git_backup.py ./backups --exclude .pyc __pycache__ .git

# 设置大小限制为50MB
python git_backup.py ./backups --size-limit 50

# 自动确认（跳过用户交互）
python git_backup.py ./backups --auto-confirm

# 不保持目录结构
python git_backup.py ./backups --no-structure

# 不创建时间戳目录
python git_backup.py ./backups --no-timestamp
```

### 2. Python代码中使用

```python
from git_backup import backup_git_tracked_files

# 基本备份
backup_dir = backup_git_tracked_files(
    backup_dir="./backups",
    preserve_structure=True,
    create_timestamp_dir=True
)

# 带排除和大小限制的备份
backup_dir = backup_git_tracked_files(
    backup_dir="./backups",
    exclude_patterns=['.pyc', '__pycache__', '.git', '.vscode'],
    preserve_structure=True,
    create_timestamp_dir=True,
    size_limit_mb=100.0,  # 100MB限制
    auto_confirm=False    # 需要用户确认
)
```

## 参数说明

- `backup_dir`: 备份目标目录
- `exclude_patterns`: 要排除的文件模式列表
- `preserve_structure`: 是否保持目录结构（默认True）
- `create_timestamp_dir`: 是否创建时间戳目录（默认True）
- `size_limit_mb`: 文件大小限制，超过此值需要用户确认（默认100MB）
- `auto_confirm`: 是否自动确认，跳过用户交互（默认False）

## 输出信息

备份完成后会显示：
- 文件总数和总大小
- 成功复制的文件数量
- 失败的文件列表（如果有）
- 备份目录路径
- 详细的备份信息文件（backup_info.txt）

## 示例

查看 `backup_example.py` 和 `test_backup.py` 文件了解具体使用示例。

## 注意事项

1. 确保在git仓库根目录下运行
2. 工具会自动跳过被`.gitignore`忽略的文件
3. 如果文件大小超过限制且未设置自动确认，会提示用户确认
4. 备份信息会保存在备份目录的`backup_info.txt`文件中
