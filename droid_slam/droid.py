import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)  # 加载模型权重
        self.args = args  # 保存参数
        self.disable_vis = args.disable_vis  # 是否禁用可视化

        # 存储图像、深度、姿态、内参（在进程之间共享）
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # 过滤输入帧以确保有足够的运动
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # 前端进程
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # 后端进程
        self.backend = DroidBackend(self.net, self.video, self.args)

        # 可视化
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # 后处理 - 填充非关键帧的姿态
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ 加载训练好的模型权重 """

        print(weights)
        self.net = DroidNet()  # 初始化网络
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])  # 去掉权重字典中的“module.”前缀

        # 截取部分权重
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)  # 加载权重到网络
        self.net.to("cuda:0").eval()  # 将网络移动到GPU并设置为评估模式

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ 主线程 - 更新地图 """

        with torch.no_grad():
            # 检查是否有足够的运动
            self.filterx.track(tstamp, image, depth, intrinsics)

            # 局部捆绑调整
            self.frontend()

            # 全局捆绑调整
            # self.backend()

    def terminate(self, stream=None):
        """ 终止可视化进程，返回姿态 [t, q] """

        del self.frontend  # 删除前端进程

        torch.cuda.empty_cache()  # 清空GPU缓存
        print("#" * 32)
        self.backend(7)  # 调用后端进程

        torch.cuda.empty_cache()  # 清空GPU缓存
        print("#" * 32)
        self.backend(12)  # 调用后端进程

        camera_trajectory = self.traj_filler(stream)  # 填充姿态轨迹
        return camera_trajectory.inv().data.cpu().numpy()  # 返回逆转后的姿态轨迹

