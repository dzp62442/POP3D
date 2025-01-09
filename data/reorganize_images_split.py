"""
读取 nuscenes_infos_**.pkl 文件，将所有相关图像以软链接的方式放入 nuscenes/images/** 文件夹下
用于 MaskCLIP 提取特征时，仅提取部分子数据集的特征，以节省空间
MaskCLIP 数据集配置文件：POP3D/MaskCLIP/configs/_base_/datasets/nuscenes_trainvaltest.py
"""

import os
import pickle
import shutil
from tqdm import tqdm

ORIGIN_DATA_ROOT = "/home/B_UserData/dongzhipeng/Datasets/nuScenes"
PKL_FILE = "nuscenes_infos_val_mini.pkl"
TARGET_IMG_DIR = "nuscenes/images/val/"

# 加载pkl文件
with open(PKL_FILE, 'rb') as f:
    datas = pickle.load(f)
cam_names = list(datas[0]['cams'].keys())  # 获取相机名称列表
print(cam_names)

# 创建目标文件夹，如果不存在的话
for cam in cam_names:
    os.makedirs(os.path.join(TARGET_IMG_DIR, cam), exist_ok=True)

# 遍历数据，创建软链接
for data in tqdm(datas):
    for cam in cam_names:
        image_path = data['cams'][cam]['data_path']  # 注意 image_path 是相对项目根目录的路径，不能直接读取
        image_name = os.path.basename(image_path)
        middle_dir = image_path.split('/')[3]  # samples / sweeps
        origin_path = os.path.join(ORIGIN_DATA_ROOT, middle_dir, cam, image_name)
        target_path = os.path.join(TARGET_IMG_DIR, cam, image_name)
        # shutil.copy(origin_path, target_path)
        os.symlink(origin_path, target_path)
    
    