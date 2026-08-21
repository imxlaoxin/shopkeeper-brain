import os
import subprocess
from pathlib import Path

# 基础用法：run() - 同步执行，等待完成
os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'#设置MinerU的模型来源 ModelScope（阿里魔搭）
os.environ['MODELSCOPE_OFFLINE'] = '1'#1 表示开启离线模式（MODELSCOPE_CACHE 指定的目录）
os.environ['HF_HUB_OFFLINE'] = '1'#启用 HuggingFace Hub 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'#启用 Transformers 库离线模式

print("✅ 环境变量已配置")
print(f"   MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
print(f"   MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
print()

result = subprocess.run(
    [
        "mineru",
        "-p",
        r"G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\doc\万用表RS-12的使用.pdf",
        "-o",
        r"G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\temp_dir",
        # "--backend",
        # "pipeline"
    ],
    capture_output=True,  # 捕获输出
    encoding='utf-8',  # 指定 UTF-8 编码（Windows 下避免 GBK 问题）
    errors='replace',  # 遇到无法解码的字符用  替换
    check=False  # 不自动抛异常，手动检查
)

print(f'result: {result}')