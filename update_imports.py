#!/usr/bin/env python3
"""
脚本用于批量将 vnpy 导入引用替换为 apilot
"""
import os
import re
from pathlib import Path

def update_file(file_path):
    """更新文件中的导入语句，将 vnpy 替换为 apilot"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换导入语句
    updated_content = re.sub(r'from vnpy\.', 'from apilot.', content)
    updated_content = re.sub(r'import vnpy\.', 'import apilot.', updated_content)
    
    # 替换其他相关引用 (比如在字符串中)
    updated_content = re.sub(r'vnpy_ctastrategy', 'apilot_ctastrategy', updated_content)
    updated_content = re.sub(r'vnpy_spreadtrading', 'apilot_spreadtrading', updated_content)
    updated_content = re.sub(r'vnpy_portfoliostrategy', 'apilot_portfoliostrategy', updated_content)
    
    # 仅当有变更时才写入文件
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return True
    return False

def process_directory(directory):
    """处理目录中的所有Python文件"""
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_file(file_path):
                    print(f"已更新: {file_path}")
                    count += 1
    return count

if __name__ == "__main__":
    # 指定要处理的目录
    project_root = Path("/Users/bobbyding/Documents/GitHub/apilot")
    
    # 更新 apilot 模块
    count1 = process_directory(project_root / "apilot")
    print(f"在 apilot 模块中更新了 {count1} 个文件")
    
    # 更新示例目录
    count2 = process_directory(project_root / "examples")
    print(f"在 examples 目录中更新了 {count2} 个文件")
    
    # 更新测试目录
    count3 = process_directory(project_root / "tests")
    print(f"在 tests 目录中更新了 {count3} 个文件")
    
    # 更新 ap_mongodb 目录
    count4 = process_directory(project_root / "ap_mongodb")
    print(f"在 ap_mongodb 目录中更新了 {count4} 个文件")
    
    print(f"总共更新了 {count1 + count2 + count3 + count4} 个文件")
