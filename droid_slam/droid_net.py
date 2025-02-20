import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def cvx_upsample(data, mask):
    """ 上采样像素级变换场 """
    batch, ht, wd, dim = data.shape  # 获取数据的形状
    data = data.permute(0, 3, 1, 2)  # 调整数据维度顺序
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)  # 调整掩码的形状
    mask = torch.softmax(mask, dim=2)  # 对掩码进行softmax操作

    up_data = F.unfold(data, [3,3], padding=1)  # 展开数据
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)  # 调整展开后数据的形状

    up_data = torch.sum(mask * up_data, dim=2)  # 按掩码加权求和
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)  # 调整数据维度顺序
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)  # 调整数据形状

    return up_data  # 返回上采样后的数据

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape  # 获取视差图的形状
    disp = disp.view(batch*num, ht, wd, 1)  # 调整视差图的形状
    mask = mask.view(batch*num, -1, ht, wd)  # 调整掩码的形状
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)  # 上采样视差图并调整形状


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)  # 定义第一个卷积层
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)  # 定义第二个卷积层
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),  # 定义卷积层
            GradientClip(),  # 定义梯度裁剪层
            nn.Softplus())  # 定义Softplus激活函数

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))  # 定义上采样掩码的卷积层

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape  # 获取输入的形状
        net = net.view(batch*num, ch, ht, wd)  # 调整输入的形状

        _, ix = torch.unique(ii, return_inverse=True)  # 获取唯一索引
        net = self.relu(self.conv1(net))  # 通过第一个卷积层和ReLU激活函数

        net = net.view(batch, num, 128, ht, wd)  # 调整形状
        net = scatter_mean(net, ix, dim=1)  # 计算均值
        net = net.view(-1, 128, ht, wd)  # 调整形状

        net = self.relu(self.conv2(net))  # 通过第二个卷积层和ReLU激活函数

        eta = self.eta(net).view(batch, -1, ht, wd)  # 计算eta
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)  # 计算上采样掩码

        return .01 * eta, upmask  # 返回eta和上采样掩码


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2  # 计算相关平面的数量

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),  # 定义卷积层
            nn.ReLU(inplace=True),  # 定义ReLU激活函数
            nn.Conv2d(128, 128, 3, padding=1),  # 定义卷积层
            nn.ReLU(inplace=True))  # 定义ReLU激活函数

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),  # 定义卷积层
            nn.ReLU(inplace=True),  # 定义ReLU激活函数
            nn.Conv2d(128, 64, 3, padding=1),  # 定义卷积层
            nn.ReLU(inplace=True))  # 定义ReLU激活函数

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 定义卷积层
            nn.ReLU(inplace=True),  # 定义ReLU激活函数
            nn.Conv2d(128, 2, 3, padding=1),  # 定义卷积层
            GradientClip(),  # 定义梯度裁剪层
            nn.Sigmoid())  # 定义Sigmoid激活函数

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 定义卷积层
            nn.ReLU(inplace=True),  # 定义ReLU激活函数
            nn.Conv2d(128, 2, 3, padding=1),  # 定义卷积层
            GradientClip())  # 定义梯度裁剪层

        self.gru = ConvGRU(128, 128+128+64)  # 定义卷积GRU层
        self.agg = GraphAgg()  # 定义图聚合层

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM 更新操作 """

        batch, num, ch, ht, wd = net.shape  # 获取输入的形状

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)  # 如果没有流，则初始化为零

        output_dim = (batch, num, -1, ht, wd)  # 定义输出维度
        net = net.view(batch*num, -1, ht, wd)  # 调整形状
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)  # 通过相关编码器
        flow = self.flow_encoder(flow)  # 通过流编码器
        net = self.gru(net, inp, corr, flow)  # 通过GRU层

        ### 更新变量 ###
        delta = self.delta(net).view(*output_dim)  # 计算delta
        weight = self.weight(net).view(*output_dim)  # 计算权重

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()  # 调整delta的形状
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()  # 调整权重的形状

        net = net.view(*output_dim)  # 调整网络的形状

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))  # 计算eta和上采样掩码
            return net, delta, weight, eta, upmask  # 返回结果

        else:
            return net, delta, weight  # 返回结果


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')  # 定义特征提取网络
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')  # 定义上下文网络
        self.update = UpdateModule()  # 定义更新模块


    def extract_features(self, images):
        """ 运行特征提取网络 """

        # 归一化图像
        images = images[:, :, [2,1,0]] / 255.0  # 调整图像通道顺序并归一化
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)  # 定义均值
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)  # 定义标准差
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])  # 标准化图像

        fmaps = self.fnet(images)  # 提取特征图
        net = self.cnet(images)  # 提取上下文特征
        
        net, inp = net.split([128,128], dim=2)  # 分割上下文特征
        net = torch.tanh(net)  # 对网络特征应用tanh激活函数
        inp = torch.relu(inp)  # 对输入特征应用ReLU激活函数
        return fmaps, net, inp  # 返回特征图、网络特征和输入特征


    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """ 估计帧间的SE3或Sim3变换 """

        u = keyframe_indicies(graph)  # 获取关键帧索引
        ii, jj, kk = graph_to_edge_list(graph)  # 将图转换为边列表

        ii = ii.to(device=images.device, dtype=torch.long)  # 将ii转换为长整型
        jj = jj.to(device=images.device, dtype=torch.long)  # 将jj转换为长整型

        fmaps, net, inp = self.extract_features(images)  # 提取特征
        net, inp = net[:,ii], inp[:,ii]  # 获取对应索引的特征
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)  # 定义相关块

        ht, wd = images.shape[-2:]  # 获取图像的高度和宽度
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)  # 生成坐标网格
        
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)  # 进行投影变换
        target = coords1.clone()  # 克隆坐标

        Gs_list, disp_list, residual_list = [], [], []  # 初始化结果列表
        for step in range(num_steps):
            Gs = Gs.detach()  # 分离Gs
            disps = disps.detach()  # 分离视差图
            coords1 = coords1.detach()  # 分离坐标
            target = target.detach()  # 分离目标

            # 提取运动特征
            corr = corr_fn(coords1)  # 计算相关
            resd = target - coords1  # 计算残差
            flow = coords1 - coords0  # 计算光流

            motion = torch.cat([flow, resd], dim=-1)  # 拼接运动特征
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)  # 调整形状并裁剪

            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj)  # 更新网络

            target = coords1 + delta  # 更新目标

            for i in range(2):
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)  # 进行BA优化

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)  # 进行投影变换
            residual = (target - coords1)  # 计算残差

            Gs_list.append(Gs)  # 添加Gs到列表
            disp_list.append(upsample_disp(disps, upmask))  # 添加上采样后的视差图到列表
            residual_list.append(valid_mask * residual)  # 添加残差到列表

        return Gs_list, disp_list, residual_list  # 返回结果
