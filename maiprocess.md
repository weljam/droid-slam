整个项目的运行流程如下:

demo.py:
输入参数  图像文件路径(必须)+相机内参(必须)+可选参数

1.读取参数
2.初始化droid对象
    1.加载权重
        初始化网络
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')  # 定义特征提取网络
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')  # 定义上下文网络
        self.update = UpdateModule()  # 定义更新模块
        相关性编码器 (corr_encoder):
            两个卷积层和 ReLU 激活函数。
            输入通道数为 cor_planes，输出通道数为 128。
            第一个卷积层的核大小为 1，第二个卷积层的核大小为 3。
        流编码器 (flow_encoder):
            两个卷积层和 ReLU 激活函数。
            输入通道数为 4，输出通道数为 128 和 64。
            第一个卷积层的核大小为 7，第二个卷积层的核大小为 3。
        权重计算器 (weight):
            两个卷积层、ReLU 激活函数、梯度裁剪层和 Sigmoid 激活函数。
            输入通道数为 128，输出通道数为 2。
            卷积层的核大小均为 3。
        更新量计算器 (delta):
            两个卷积层、ReLU 激活函数和梯度裁剪层。
            输入通道数为 128，输出通道数为 2。
            卷积层的核大小均为 3。
        卷积 GRU (gru):
            输入通道数为 128，输出通道数为 128 + 128 + 64。
        图聚合层 (agg):
            包含两个卷积层和 ReLU 激活函数。
            计算 eta 和 upmask。
        UpdateModule
        ├── corr_encoder
        │   ├── Conv2d(cor_planes, 128, kernel_size=1)
        │   ├── ReLU(inplace=True)
        │   ├── Conv2d(128, 128, kernel_size=3, padding=1)
        │   ├── ReLU(inplace=True)
        ├── flow_encoder
        │   ├── Conv2d(4, 128, kernel_size=7, padding=3)
        │   ├── ReLU(inplace=True)
        │   ├── Conv2d(128, 64, kernel_size=3, padding=1)
        │   ├── ReLU(inplace=True)
        ├── weight
        │   ├── Conv2d(128, 128, kernel_size=3, padding=1)
        │   ├── ReLU(inplace=True)
        │   ├── Conv2d(128, 2, kernel_size=3, padding=1)
        │   ├── GradientClip()
        │   ├── Sigmoid()
        ├── delta
        │   ├── Conv2d(128, 128, kernel_size=3, padding=1)
        │   ├── ReLU(inplace=True)
        │   ├── Conv2d(128, 2, kernel_size=3, padding=1)
        │   ├── GradientClip()
        ├── gru
        │   ├── ConvGRU(128, 128+128+64)
        ├── agg
        │   ├── GraphAgg()
    2.初始化DepthVideo(args.image_size, args.buffer, stereo=args.stereo)
        1.获得当前帧的数量,图像长宽
        2.初始化图像帧的对应参数和属性,特征图,网络等
    3.初始化MotionFilter(self.net, self.video, thresh=args.filter_thresh)
        1.设置网络,上下文网络,特征网络,更新网络.
        2.保存图像帧率,并进行图像的均真标准差计算
    4.
