import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # 全局优化窗口
        self.t0 = 0
        self.t1 = 0

        # 参数初始化
        self.upsample = args.upsample  # 是否进行上采样
        self.beta = args.beta  # 优化中的超参数 beta
        self.backend_thresh = args.backend_thresh  # 后端处理的阈值
        self.backend_radius = args.backend_radius  # 后端处理的半径
        self.backend_nms = args.backend_nms  # 后端处理的非极大值抑制参数
        
    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value
        if not self.video.stereo and not torch.any(self.video.disps_sens):
             self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t, upsample=self.upsample)

        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True
