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
from plyfile import PlyData

def read_lasfile(file_path):
    with laspy.open(file_path) as fh:
        # print('Points from Header:', fh.header.point_count)
        las = fh.read()
        
        # 0 Unclassified        -> invaild
        # 1 Low-vegetation      -> 1 Low-vegetation
        # 2 Terrain             -> 0 Terrain
        # 3 Out-points          -> invaild
        # 4 Stem                -> 2 Wood
        # 5 Live branches       -> 3 Leaf
        # 6 Woody branches      -> 2 Wood
        classification_mapping = {
            1: 1,    # Low-vegetation -> 1
            2: 0,    # Terrain -> 0
            4: 2,    # Stem -> 2
            5: 4,    # Live branches -> 3
            6: 3     # Woody branches -> 2
        }
        original_class = las.classification
        
        converted_label = np.full_like(original_class, -1, dtype=int)
        for orig_code, target_label in classification_mapping.items():
            converted_label[original_class == orig_code] = target_label
        
        valid_mask = converted_label != -1  # 只保留有效映射的点
        xyz = np.vstack((las.x[valid_mask], 
                         las.y[valid_mask], 
                         las.z[valid_mask])).transpose()
        label = converted_label[valid_mask]

    return xyz, label


def read_plyfile(file_path):
    # 读取 PLY 文件
    ply = PlyData.read(file_path)
    # 访问顶点和面数据
    coords  = np.vstack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']]).T
    segment = np.array(ply['vertex']['semantic_seg']).astype(np.int32)

    coords = coords - coords.mean(0)
    # exit()

    ##
    idx = np.where(segment > 0)
    segment = segment[idx]
    coords = coords[idx]

    segment = segment - 1

    print(np.unique(segment))
    # exit()

    return coords, segment

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
    dataset_root, scene_path, output_path, scene_paths
):
    if 'train' in str(scene_path):
        output_path = os.path.join(output_path, "train", f"{scene_path.name}")
        split_name = "train"
    elif 'test' in str(scene_path):
        output_path = os.path.join(output_path, "test", f"{scene_path.name}")
        split_name = "test"
    elif 'val' in str(scene_path):
        output_path = os.path.join(output_path, "val", f"{scene_path.name}")
        split_name = "val"
    else:
        pass

    print(f"Processing: {scene_path} in {split_name}")

    # coords, segment = read_lasfile(Path(dataset_root, "raw", scene_path))
    coords, segment = read_plyfile(scene_path)

    # 防止精度丢失
    coords = coords - coords.min(0)

    unique_labels = np.unique(segment)
    counts = np.bincount(segment)
    result = {int(lbl): int(counts[lbl]) for lbl in unique_labels}

    save_dict = dict(
        coord=coords.astype(np.float32),
        segment=segment.astype(np.uint8),
    )

    # print(np.unique(segment))
    unique_labels = np.unique(segment)
    counts = np.bincount(segment)
    result = {int(lbl): int(counts[lbl]) for lbl in unique_labels}
    save_to_txt(coords, segment.reshape(-1, 1), scene_path.name + ".txt")

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    for key in save_dict.keys():
        np.save(os.path.join(output_path, f"{key}.npy"), save_dict[key])

    return result


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
    # df = pd.read_csv(
    #     Path(config.dataset_root) / "data_split_metadata.csv",
    # )

    # dev_paths = df[df['split'] == 'dev']['path'].tolist()
    # test_paths = df[df['split'] == 'test']['path'].tolist()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = Path(config.dataset_root).glob("*/*/*.ply")
    scene_paths = [name for name in scene_paths]
    scene_paths = sorted(scene_paths)

    total = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for fn in scene_paths:
        nums = handle_process(config.dataset_root, fn, config.output_root, scene_paths)

        for key in nums.keys():
            total[key] = total[key] + nums[key]
    print(total)
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
