import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid_slam.droid import Droid

import torch.nn.functional as F


def show_image(image):
    # 将图像从张量转换为numpy数组并显示
    image = image.permute(1, 2, 0).cpu().numpy()  # 调整图像维度并转换为numpy数组
    cv2.imshow('image', image / 255.0)  # 显示图像
    cv2.waitKey(1)  # 等待键盘输入

def image_stream(imagedir, calib, stride):
    """ 图像生成器 """

    # 读取校准文件
    calib = np.loadtxt(calib, delimiter=" ")  # 加载校准参数
    fx, fy, cx, cy = calib[:4]  # 提取内参

    # 构建内参矩阵
    K = np.eye(3)  # 创建3x3单位矩阵
    K[0,0] = fx  # 设置焦距fx
    K[0,2] = cx  # 设置光心cx
    K[1,1] = fy  # 设置焦距fy
    K[1,2] = cy  # 设置光心cy

    # 获取图像列表并按步长筛选
    image_list = sorted(os.listdir(imagedir))[::stride]  # 获取图像文件列表并按步长筛选

    for t, imfile in enumerate(image_list):
        # 读取图像文件
        image = cv2.imread(os.path.join(imagedir, imfile))  # 读取图像
        if len(calib) > 4:
            # 如果校准参数包含畸变系数，则进行去畸变
            image = cv2.undistort(image, K, calib[4:])  # 去畸变

        # 调整图像大小
        h0, w0, _ = image.shape  # 获取原始图像尺寸
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))  # 计算调整后的高度
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))  # 计算调整后的宽度

        image = cv2.resize(image, (w1, h1))  # 调整图像大小
        image = image[:h1-h1%8, :w1-w1%8]  # 裁剪图像使其尺寸为8的倍数
        image = torch.as_tensor(image).permute(2, 0, 1)  # 转换为张量并调整维度

        # 调整内参
        intrinsics = torch.as_tensor([fx, fy, cx, cy])  # 创建内参张量
        intrinsics[0::2] *= (w1 / w0)  # 调整fx和cx
        intrinsics[1::2] *= (h1 / h0)  # 调整fy和cy

        yield t, image[None], intrinsics  # 生成图像和内参

def save_reconstruction(droid, reconstruction_path):
    # 保存重建结果

    from pathlib import Path
    import random
    import string

    # 获取视频数据
    t = droid.video.counter.value  # 获取视频帧数
    tstamps = droid.video.tstamp[:t].cpu().numpy()  # 获取时间戳
    images = droid.video.images[:t].cpu().numpy()  # 获取图像
    disps = droid.video.disps_up[:t].cpu().numpy()  # 获取深度图
    poses = droid.video.poses[:t].cpu().numpy()  # 获取位姿
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()  # 获取内参

    # 创建保存路径
    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)  # 创建目录
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)  # 保存时间戳
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)  # 保存图像
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)  # 保存深度图
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)  # 保存位姿
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)  # 保存内参

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")  # 图像目录路径
    parser.add_argument("--calib", type=str, help="path to calibration file")  # 校准文件路径
    parser.add_argument("--t0", default=0, type=int, help="starting frame")  # 起始帧
    parser.add_argument("--stride", default=3, type=int, help="frame stride")  # 帧步长

    parser.add_argument("--weights", default="droid.pth")  # 权重文件路径
    parser.add_argument("--buffer", type=int, default=512)  # 缓冲区大小
    parser.add_argument("--image_size", default=[240, 320])  # 图像大小
    parser.add_argument("--disable_vis", action="store_true")  # 禁用可视化

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")  # 平移/旋转分量的权重
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")  # 运动阈值
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")  # 预热帧数
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")  # 新关键帧的阈值
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")  # 前端阈值
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")  # 前端优化窗口
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")  # 前端半径
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")  # 非极大值抑制

    parser.add_argument("--backend_thresh", type=float, default=22.0)  # 后端阈值
    parser.add_argument("--backend_radius", type=int, default=2)  # 后端半径
    parser.add_argument("--backend_nms", type=int, default=3)  # 后端非极大值抑制
    parser.add_argument("--upsample", action="store_true")  # 是否上采样
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")  # 重建结果保存路径
    args = parser.parse_args()

    args.stereo = False  # 立体视觉标志
    torch.multiprocessing.set_start_method('spawn')  # 设置多进程启动方法

    droid = None  # 初始化Droid对象

    # 需要高分辨率深度图
    if args.reconstruction_path is not None:
        args.upsample = True  # 如果指定了重建路径，则启用上采样

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue  # 跳过起始帧之前的帧

        if not args.disable_vis:
            show_image(image[0])  # 显示图像

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]  # 设置图像大小
            droid = Droid(args)  # 初始化Droid对象
        
        droid.track(t, image, intrinsics=intrinsics)  # 跟踪图像

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)  # 保存重建结果

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))  # 终止Droid并获取轨迹估计
