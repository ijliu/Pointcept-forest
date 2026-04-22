"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import argparse
import json
import laspy
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from scipy.spatial import KDTree

def read_lasfile(file_path):
    las = laspy.read(file_path)
    xyz = las.xyz
    return xyz


def save_to_txt(xyz, label, save_path):
    """将xyz和label合并并保存为TXT文件"""
    # 合并xyz（N,3）和label（N,1）为（N,4）的数组
    data = np.hstack((xyz, label))
    
    # 保存为TXT，格式：x y z label（空格分隔）
    np.savetxt(
        save_path,
        data,
        fmt='%.6f %.6f %.6f %d',  # xyz保留6位小数，label为整数
        delimiter=' '  # 空格分隔
    )
    print(f"已保存到: {save_path}")

def handle_process(
    dataset_root, scene_path, output_path
):
    scene_id = Path(scene_path).parts[-2]
    output_path = os.path.join(output_path, "test", f"{scene_id}")
    split_name = "test"
    
    print(f"Processing: {scene_id} in {split_name}")

    coords = read_lasfile(Path(dataset_root, scene_path))

    # 防止精度丢失
    coords = coords - coords.mean(0)
    segment = np.zeros((coords.shape[0]))
    save_dict = dict(
        coord=coords.astype(np.float32),
        segment=segment.astype(np.float32)
    )

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    for key in save_dict.keys():
        np.save(os.path.join(output_path, f"{key}.npy"), save_dict[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )

    config = parser.parse_args()
    
    scene_paths = Path(config.dataset_root).glob("*/*ULS*")

    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    for fn in scene_paths:
        handle_process(config.dataset_root, fn, config.output_root)
    exit()

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            handle_process,
            repeat(config.dataset_root),
            scene_paths,
            repeat(config.output_root),
            repeat(dev_paths),
            repeat(test_paths)
        )
    )
