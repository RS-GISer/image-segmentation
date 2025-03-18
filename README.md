# image-segmentation

## News

- 🤗 更新训练配置文件介绍
- 🤗 更新mmsegmentation环境配置和测试

这个项目基于mmsegmentation框架(mmlab)，
提供mmsegmentation从0到1的实践，包含训练和推理教程

# Prepare 

我们使用了mmsegmentation的分支：mmsegmenation-1.x

[**mmsegmentation-1.x(github)**](https://github.com/open-mmlab/mmsegmentation/tree/1.x)

Clone mmsegmentation-1.x:
```
git clone https://github.com/open-mmlab/mmsegmentation.git
```

## 创建环境

测试环境（推荐）：
```
Python 3.8.0
pytorch 2.0.0
torchvision 0.15.0
mmcv 2.1.0
mmdeploy  1.3.1
mmengine  0.10.3
mmsegmentation  1.2.2
```
参考：MMSegmentation 官方文档：[**mmsegmentation官方文档**](https://mmsegmentation.readthedocs.io/zh-cn/latest/get_started.html)</br>
pytorch官网(下载):[**Previous PyTorch Versions**](https://pytorch.org/get-started/previous-versions/)

**创建环境（需要安装anaconda/miniconda）**
```
conda create --name mmseg python=3.8 -y   # 创建环境
conda activate mmseg  # 激活环境
```
**安装 pytoch**
``` 
# 官方安装pytorch
conda install pytorch torchvision -c pytorch  
# 推荐
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
``` 

需要安装和torch、cuda对应版本的mmcv：[**mmcv**](https://mmcv.readthedocs.io/en/latest/get_started/installation.html )

<p align="center">
  <img src="./imgs/mmcv.png" style="width:90%;">
</p>

```
# install mmcv
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
**other need**
``` 
pip install -U openmim
pip install mmdeploy==1.3.1
pip install mmengine==0.10.3
``` 
install mmsegmentation：Installation from source code/Use mmsegmentation as a third-party library

安装mmsegmentation：源码安装/第三方库安装
- Installation from source code(源码安装)：直接在当前项目中开发和运行mmsegmentation
- Use mmsegmentation as a third-party library(第三方库安装：mmsegmentation作为依赖库，如果需要修改网络结构、添加数据集，需要到conda环境修改（mmseg\Lib\site-packages\mmseg）
``` 
# Installation from source code 源码安装
cd mmsegmentation
pip install -v -e .
# Use mmsegmentation as a third-party library  第三方库安装
pip install mmsegmentation==1.2.2
``` 

## 验证环境
download config and checkpoints in：
``` 
cd .\mmsegmentation-1.x\
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
``` 
可以在当前工作目录看到 **pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py** | **pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth**

## 测试
``` 
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```
你将在当前文件夹中看到一个新图像result.jpg
<p align="center">
  <img src="./imgs/pspnet.png" style="width:90%;">
</p>

# 如何训练

使用mmsegmentation-1.x/tools/train.py进行图像分割模型训练

## 配置文件介绍

- mmsegmentation-1.x/configs文件夹中为训练配置文件，其中_base_下下面有4种基本组件类型： 数据集(dataset)，模型(model)，
训练策略(schedule)和运行时的默认设置(default runtime)

- 选择upernet网络作为训练测试配置网络，在mmsegmentation-1.x/configs/upernet中选择合适的网络结构，
例如：mmsegmentation-1.x\configs\upernet\upernet_r50_4xb4-80k_ade20k-512x512.py，其中

- upernet为网络名称，r50表示骨干网络，4xb4一般表示网络大小，160k表示训练策略（160k的iters），ade20k为数据集，512*512为图像尺寸

- upernet_r50_4xb4-80k_ade20k-512x512.py文件中_base_=[]，表示其继承的配置信息， 在upernet_r50_4xb4-80k_ade20k-512x512.py修改文件会覆盖继承的文件

### 配置文件参数介绍

- upernet_r50_4xb4-80k_ade20k-512x512.py
  - crop_size：裁切图像尺寸
  - decode_head：解码器头，num_classes=150 表明解码器头要预测的类别数量为 150 个
  - auxiliary_head：辅助解码器头，num_classes=150 表明辅助解码器头要预测的类别数量为 150 个


- 网络设置：../_base_/models/upernet_r50.py
  - norm_cfg
    - type='SyncBN'：指定使用同步批量归一化（Synchronized Batch Normalization）
    - requires_grad=True：表示该归一化层的参数需要进行梯度更新，即参与模型的训练过程
  - data_preprocessor
    - type='SegDataPreProcessor'：使用 MMSegmentation 中的 SegDataPreProcessor 类进行数据预处理
    - mean/std：图像三个通道的均值和标准差，用于对输入图像进行归一化操作。
    - bgr_to_rgb=True：将输入图像的通道顺序从 BGR 转换为 RGB
    - pad_val=0：图像填充值，当对图像进行填充操作时，使用 0 进行填充
    - seg_pad_val=255：分割标签的填充值，用于在填充分割标签时使用 255 作为填充值
  - model
    - type='EncoderDecoder'：指定模型类型为编码器 - 解码器结构
    - data_preprocessor=data_preprocessor：将前面定义的数据预处理配置应用到模型中
    - pretrained='open-mmlab://resnet50_v1c'：使用预训练的 ResNet-50 v1c 模型作为骨干网络的初始化权重，**设置权重路径**
    - backbone：骨干网络设置。设置Resnet网络深度、每个阶段的膨胀率、每个阶段的步长等等
    - decode_head：解码器头配置。UPerHead指定解码器头使用 UPerHead 架构，设置输入特征图的通道数、金字塔池化的尺度、使用前面定义的归一化配置、解码器头的损失函数等
    - auxiliary_head：辅助解码器头配置。辅助解码器头使用 FCNHead 架构，设置输入特征图的通道数、输入特征图的通道数、损失函数等


- 数据集设置：../_base_/datasets/ade20k.py
  - dataset_type = 'ADE20KDataset'：指定使用的数据集类型为 ADE20KDataset
  - data_root = 'data/ade/ADEChallengeData2016'：指定数据集的根目录
  - crop_size = (512, 512)：定义裁剪图像的大小
  - train_pipeline
    - type='LoadImageFromFile'：从文件中加载图像数据 
    - type='LoadAnnotations'：reduce_zero_label=True：加载图像对应的标注信息，reduce_zero_label=True 表示将标注中的类别 0 视为背景，并且将其标签值减去 1，使得类别标签从 0 开始连续编号。**二分类必须修改为False**
    - type='RandomResize'：随机调整图像的大小，scale=(2048, 512) 表示缩放的参考尺寸，ratio_range=(0.5, 2.0) 表示缩放比例的随机范围，keep_ratio=True 表示保持图像的宽高比。 
    - type='RandomCrop'：随机裁剪图像，裁剪尺寸为 crop_size，cat_max_ratio=0.75 表示裁剪区域内同一类别的像素占比不能超过 0.75，以保证裁剪后的图像具有多样性。 
    - type='RandomFlip'：prob=0.5：以 0.5 的概率随机翻转图像
    - type='PhotoMetricDistortion'：对图像进行光度失真处理，包括亮度、对比度、饱和度和色调的随机调整
    - type='PackSegInputs'：将处理后的图像和标注信息打包成模型可以接受的输入格式
  - test_pipeline
    - type='LoadImageFromFile'：从文件中加载图像数据
    - type='Resize'：将图像调整为指定的大小，保持宽高比
    - type='LoadAnnotations'：加载图像对应的标注信息，并对标签进行处理。
    - type='PackSegInputs'：将处理后的图像和标注信息打包成模型可以接受的输入格式
  - tta_pipeline：测试时增强（TTA）数据处理流程
    - img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]：缩放比例用于测试时的多尺度增强。
    - dict(type='LoadImageFromFile', backend_args=None)：从文件中加载图像数据。
    - dict(type='TestTimeAug', ...)：进行测试时增强（TTA），通过对图像进行不同尺度的缩放和翻转操作、加载标注信息
  - train_dataloader：训练数据加载器
    - batch_size=4：每批次样本个数 4
    - num_workers=4：进程数 4
    - persistent_workers=True：数据加载完成后不被销毁
    - sampler=dict(type='InfiniteSampler', shuffle=True)：使用 InfiniteSampler 采样器，该采样器可以无限循环地提供样本，shuffle=True 表示在每个 epoch 开始时对样本进行打乱。 
    - dataset：**指定数据集图像和标签路径**，定义数据集的相关信息，包括数据集类型、数据根目录、数据前缀（指定图像和标注文件的子目录）以及使用的处理流程（train_pipeline）
  - val_dataloader：训练数据加载器 
    - batch_size=1：每批次样本个数 1
    - num_workers=4：进程数 4 
    - persistent_workers=True：数据加载完成后不被销毁
    - sampler=dict(type='DefaultSampler', shuffle=False)：使用 DefaultSampler 采样器，shuffle=False 表示在验证过程中不打乱样本顺序。 
    - dataset：**指定数据集图像和标签路径**，定义数据集的相关信息，包括数据集类型、数据根目录、数据前缀（指定图像和标注文件的子目录）以及使用的处理流程（train_pipeline）
  - test_dataloader：测试数据加载器
    - 测试数据加载器使用与验证数据加载器相同的配置
  - val_evaluator：评估指标
    - type='IoUMetric'：使用交并比（IoU）指标进行评估
    - iou_metrics=['mIoU']：计算平均交并比（mIoU）作为评估指标


- 训练策略设置：../_base_/schedules/schedule_80k.py
  - optimizer：优化器
    - type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005：使用随机梯度下降（Stochastic Gradient Descent，SGD）作为优化算法；学习率 0.01；动量因子设置为 0.9；权重衰减系数为 0.0005
  - optim_wrapper：优化器包装器
    - type='OptimWrapper'：使用 OptimWrapper 对优化器进行包装
    - optimizer=optimizer：指定使用前面定义的优化器。 
    - clip_grad=None：不进行梯度裁剪
  - param_scheduler：学习率调度策略
    - type='PolyLR'：使用多项式学习率衰减策略（Polynomial Learning Rate Decay）。这种策略会随着训练迭代次数的增加，按照多项式函数的形式逐渐降低学习率
    - eta_min=1e-4：学习率的最小值为1e-4，当学习率衰减到这个值时，将不再继续降低
    - power=0.9：决定了学习率衰减的速度0.9
    - begin=0：学习率调度从第 0 次迭代开始生效
    - end=80000：学习率调度在第 80000 次迭代结束
    - by_epoch=False：表示学习率的调整是基于迭代次数（iterations），而不是基于训练轮数（epochs）
  - train_cfg（训练配置）
    - type='IterBasedTrainLoop'：使用基于迭代次数的训练循环，即训练过程以迭代次数为单位进行控制
    - max_iters=80000：最大训练迭代次数为 80000 次
    - val_interval=8000：每进行 8000 次迭代后进行一次验证
  - val_cfg（验证配置）：type='ValLoop'：使用 ValLoop 作为验证循环，用于在验证集上评估模型的性能
  - test_cfg（测试配置）：type='TestLoop'：使用 TestLoop 作为测试循环，用于在测试集上评估模型的最终性能
  - default_hooks：钩子 
    - timer（迭代计时器钩子）
      - type='IterTimerHook'：用于记录每次迭代的时间，方便监控训练过程的效率
      - logger（日志记录钩子）
        - type='LoggerHook'：用于记录训练过程中的各种指标和信息。
        - interval=50：每 50 次迭代记录一次日志。 
        - log_metric_by_epoch=False：表示日志记录是基于迭代次数，而不是基于训练轮数
    - param_scheduler（学习率调度钩子）
      - type='ParamSchedulerHook'：根据前面定义的学习率调度策略，在每次迭代时更新学习率
    - checkpoint（模型检查点钩子）
      - type='CheckpointHook'：保存模型权重
      - by_epoch=False：基于迭代次数保存权重 
      - interval=8000：每 8000 次迭代保存一次权重 
    - sampler_seed（分布式采样器种子钩子）
      - type='DistSamplerSeedHook'：在分布式训练中，用于设置采样器的随机种子，确保不同进程的采样顺序一致
    - visualization（可视化钩子）
      - type='SegVisualizationHook'：语义分割任务的可视化

## 训练流程

- 训练命令调用，单GPU训练
``` 
python tools/train.py  ${配置文件} [可选参数]
``` 

- 
