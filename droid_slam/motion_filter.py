import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        # 初始化MotionFilter类的实例
        # net: 包含网络模型的对象
        # video: 视频对象，用于存储处理后的帧
        # thresh: 阈值，用于判断是否有足够的运动
        # device: 设备类型，默认为"cuda:0"

        # 分离网络模块
        self.cnet = net.cnet  # 上下文网络
        self.fnet = net.fnet  # 特征网络
        self.update = net.update  # 更新网络

        self.video = video  # 视频对象
        self.thresh = thresh  # 运动阈值
        self.device = device  # 设备类型

        self.count = 0  # 计数器

        # 图像归一化的均值和标准差
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """
        # 主更新操作 - 在视频的每一帧上运行

        Id = lietorch.SE3.Identity(1,).data.squeeze()  # 获取单位变换矩阵
        ht = image.shape[-2] // 8  # 计算图像高度的八分之一
        wd = image.shape[-1] // 8  # 计算图像宽度的八分之一

        # 归一化图像
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0  # 将图像转换为张量并归一化
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)  # 减去均值并除以标准差

        # 提取特征
        gmap = self.__feature_encoder(inputs)  # 使用特征编码器提取特征

        ### 总是将第一帧添加到深度视频中 ###
        if self.video.counter.value == 0:  # 如果是第一帧
            net, inp = self.__context_encoder(inputs[:,[0]])  # 使用上下文编码器提取特征
            self.net, self.inp, self.fmap = net, inp, gmap  # 保存提取的特征
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])  # 将特征添加到视频中

        ### 只有在有足够运动的情况下才添加新帧 ###
        else:                
            # 索引相关体积
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]  # 生成坐标网格
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)  # 计算相关体积

            # 使用一次更新迭代近似流量大小
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)  # 更新网络并计算流量变化

            # 检查运动幅度 / 将新帧添加到视频中
            if delta.norm(dim=-1).mean().item() > self.thresh:  # 如果运动幅度大于阈值
                self.count = 0  # 重置计数器
                net, inp = self.__context_encoder(inputs[:,[0]])  # 使用上下文编码器提取特征
                self.net, self.inp, self.fmap = net, inp, gmap  # 保存提取的特征
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])  # 将特征添加到视频中

            else:
                self.count += 1  # 增加计数器




# class MotionFilter:
#     """ This class is used to filter incoming frames and extract features """

#     def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
#         # split net modules
#         self.cnet = net.cnet
#         self.fnet = net.fnet
#         self.update = net.update

#         self.video = video
#         self.thresh = thresh
#         self.device = device

#         self.count = 0

#         # mean, std for image normalization
#         self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
#         self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
#     @torch.cuda.amp.autocast(enabled=True)
#     def __context_encoder(self, image):
#         """ context features """
#         x = self.cnet(image)
#         net, inp = self.cnet(image).split([128,128], dim=2)
#         return net.tanh().squeeze(0), inp.relu().squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     def __feature_encoder(self, image):
#         """ features for correlation volume """
#         return self.fnet(image).squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     @torch.no_grad()
#     def track(self, tstamp, image, depth=None, intrinsics=None):
#         """ main update operation - run on every frame in video """

#         Id = lietorch.SE3.Identity(1,).data.squeeze()
#         ht = image.shape[-2] // 8
#         wd = image.shape[-1] // 8

#         # normalize images
#         inputs = image[None, None, [2,1,0]].to(self.device) / 255.0
#         inputs = inputs.sub_(self.MEAN).div_(self.STDV)

#         # extract features
#         gmap = self.__feature_encoder(inputs)

#         ### always add first frame to the depth video ###
#         if self.video.counter.value == 0:
#             net, inp = self.__context_encoder(inputs)
#             self.net, self.inp, self.fmap = net, inp, gmap
#             self.video.append(tstamp, image, Id, 1.0, intrinsics / 8.0, gmap[0], net[0], inp[0])

#         ### only add new frame if there is enough motion ###
#         else:                
#             # index correlation volume
#             coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
#             corr = CorrBlock(self.fmap[None], gmap[None])(coords0)

#             # approximate flow magnitude using 1 update iteration
#             _, delta, weight = self.update(self.net[None], self.inp[None], corr)

#             # check motion magnitue / add new frame to video
#             if delta.norm(dim=-1).mean().item() > self.thresh:
#                 self.count = 0
#                 net, inp = self.__context_encoder(inputs)
#                 self.net, self.inp, self.fmap = net, inp, gmap
#                 self.video.append(tstamp, image, None, None, intrinsics / 8.0, gmap[0], net[0], inp[0])

#             else:
#                 self.count += 1

