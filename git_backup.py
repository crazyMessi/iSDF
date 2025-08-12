#!/usr/bin/env python3
"""
Git文件备份工具
用于将当前被git追踪的文件复制到指定位置
"""

import os
import shutil
import subprocess
import datetime
from pathlib import Path
from typing import List, Optional


def backup_git_tracked_files(
    backup_dir: str,
    exclude_patterns: Optional[List[str]] = None,
    preserve_structure: bool = True,
    create_timestamp_dir: bool = True,
    size_limit_mb: float = 100.0,
    auto_confirm: bool = False
) -> str:
    """
    将当前被git追踪的文件复制到指定位置
    
    Args:
        backup_dir: 备份目标目录
        exclude_patterns: 要排除的文件模式列表，例如 ['.pyc', '__pycache__']
        preserve_structure: 是否保持目录结构
        create_timestamp_dir: 是否在备份目录下创建时间戳子目录
        size_limit_mb: 文件大小限制（MB），超过此值需要用户确认
        auto_confirm: 是否自动确认，跳过用户交互
    
    Returns:
        实际备份目录的路径
    """
    
    # 获取git仓库根目录
    try:
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=os.getcwd(),
            text=True
        ).strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("当前目录不是git仓库")
    
    # 获取被git追踪的文件列表（只包含实际被追踪的文件）
    try:
        tracked_files = subprocess.check_output(
            ['git', 'ls-files'],
            cwd=git_root,
            text=True
        ).strip().split('\n')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"获取git文件列表失败: {e}")
    
    # 过滤掉空行
    tracked_files = [f for f in tracked_files if f.strip()]
    
    # 进一步验证文件是否真的被git追踪（排除被.gitignore忽略的文件）
    verified_files = []
    for file_path in tracked_files:
        source_path = os.path.join(git_root, file_path)
        if os.path.exists(source_path):
            # 检查文件是否被git忽略
            try:
                result = subprocess.run(
                    ['git', 'check-ignore', '--quiet', file_path],
                    cwd=git_root,
                    capture_output=True
                )
                # 如果返回码为0，说明文件被忽略
                if result.returncode != 0:
                    verified_files.append(file_path)
                else:
                    print(f"跳过被忽略的文件: {file_path}")
            except subprocess.CalledProcessError:
                # 如果检查失败，保守起见包含该文件
                verified_files.append(file_path)
        else:
            print(f"警告: 文件不存在 {source_path}")
    
    tracked_files = verified_files
    
    # 应用排除模式
    if exclude_patterns:
        filtered_files = []
        for file_path in tracked_files:
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in file_path:
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_files.append(file_path)
        tracked_files = filtered_files
    
    # 统计文件大小
    total_size_bytes = 0
    file_sizes = {}
    
    for file_path in tracked_files:
        source_path = os.path.join(git_root, file_path)
        try:
            file_size = os.path.getsize(source_path)
            total_size_bytes += file_size
            file_sizes[file_path] = file_size
        except OSError as e:
            print(f"无法获取文件大小 {file_path}: {e}")
    
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    print(f"\n备份统计信息:")
    print(f"文件总数: {len(tracked_files)}")
    print(f"总大小: {total_size_mb:.2f} MB ({total_size_bytes:,} 字节)")
    
    # 如果文件大小超过限制且未设置自动确认，则询问用户
    if total_size_mb > size_limit_mb and not auto_confirm:
        print(f"\n警告: 备份文件总大小 ({total_size_mb:.2f} MB) 超过限制 ({size_limit_mb} MB)")
        response = input("是否继续备份? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("备份已取消")
            return None
    
    # 确定备份目录
    if create_timestamp_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_backup_dir = os.path.join(backup_dir, f"git_backup_{timestamp}")
    else:
        actual_backup_dir = backup_dir
    
    # 创建备份目录
    os.makedirs(actual_backup_dir, exist_ok=True)
    
    # 复制文件
    copied_count = 0
    failed_files = []
    
    for file_path in tracked_files:
        source_path = os.path.join(git_root, file_path)
        
        if not os.path.exists(source_path):
            print(f"警告: 文件不存在 {source_path}")
            continue
        
        try:
            if preserve_structure:
                # 保持目录结构
                dest_path = os.path.join(actual_backup_dir, file_path)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
            else:
                # 只复制文件名，不保持结构
                filename = os.path.basename(file_path)
                dest_path = os.path.join(actual_backup_dir, filename)
                # 如果文件名冲突，添加路径前缀
                if os.path.exists(dest_path):
                    safe_filename = file_path.replace('/', '_').replace('\\', '_')
                    dest_path = os.path.join(actual_backup_dir, safe_filename)
            
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            print(f"已复制: {file_path}")
            
        except Exception as e:
            failed_files.append((file_path, str(e)))
            print(f"复制失败: {file_path} - {e}")
    
    # 创建备份信息文件
    backup_info = {
        'backup_time': datetime.datetime.now().isoformat(),
        'git_root': git_root,
        'total_files': len(tracked_files),
        'copied_files': copied_count,
        'failed_files': failed_files,
        'backup_dir': actual_backup_dir,
        'total_size_mb': total_size_mb,
        'total_size_bytes': total_size_bytes
    }
    
    info_file = os.path.join(actual_backup_dir, 'backup_info.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("Git文件备份信息\n")
        f.write("=" * 50 + "\n")
        f.write(f"备份时间: {backup_info['backup_time']}\n")
        f.write(f"Git仓库: {backup_info['git_root']}\n")
        f.write(f"总文件数: {backup_info['total_files']}\n")
        f.write(f"总大小: {backup_info['total_size_mb']:.2f} MB\n")
        f.write(f"成功复制: {backup_info['copied_files']}\n")
        f.write(f"失败文件: {len(backup_info['failed_files'])}\n")
        f.write(f"备份目录: {backup_info['backup_dir']}\n")
        
        if failed_files:
            f.write("\n失败文件列表:\n")
            for file_path, error in failed_files:
                f.write(f"  {file_path}: {error}\n")
    
    print(f"\n备份完成!")
    print(f"备份目录: {actual_backup_dir}")
    print(f"成功复制: {copied_count}/{len(tracked_files)} 个文件")
    print(f"总大小: {total_size_mb:.2f} MB")
    if failed_files:
        print(f"失败文件: {len(failed_files)} 个")
    
    return actual_backup_dir


def main():
    """主函数，演示如何使用备份功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Git文件备份工具')
    parser.add_argument('backup_dir', help='备份目标目录')
    parser.add_argument('--exclude', nargs='*', help='要排除的文件模式')
    parser.add_argument('--no-structure', action='store_true', help='不保持目录结构')
    parser.add_argument('--no-timestamp', action='store_true', help='不创建时间戳目录')
    parser.add_argument('--size-limit', type=float, default=100.0, help='文件大小限制（MB），默认100MB')
    parser.add_argument('--auto-confirm', action='store_true', help='自动确认，跳过用户交互')
    
    args = parser.parse_args()
    
    try:
        backup_dir = backup_git_tracked_files(
            backup_dir=args.backup_dir,
            exclude_patterns=args.exclude,
            preserve_structure=not args.no_structure,
            create_timestamp_dir=not args.no_timestamp,
            size_limit_mb=args.size_limit,
            auto_confirm=args.auto_confirm
        )
        print(f"\n备份已保存到: {backup_dir}")
    except Exception as e:
        print(f"备份失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
