# image-segmentation

## News
- 🤗 updating the mmsegmentation environment configuration and environment test(更新mmsegmentation环境配置和测试)

## description
This project is based on the mmsegmentation framework (mmlab).
Provides mmsegmentation practice from 0 to 1, including training and reasoning tutorials
</br>这个项目基于mmsegmentation框架(mmlab)，
提供mmsegmentation从0到1的实践，包含训练和推理教程

mmsegmentation-1.x: [**github**](https://github.com/open-mmlab/mmsegmentation/tree/1.x)

## prepare

The branch of mmsegmentation we use is mmsegmenation-1.x

我们使用了mmsegmentation的分支：mmsegmenation-1.x

Clone mmsegmentation-1.x:
```
git clone https://github.com/open-mmlab/mmsegmentation.git
```

## Create Environment
Testing Environment (Recommended)</br>
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

create environment
```
conda create --name mmseg python=3.8 -y   # 创建环境
conda activate mmseg  # 激活环境
```
**install pytoch**
``` 
# 官方安装pytorch
conda install pytorch torchvision -c pytorch  
# 推荐
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
``` 
It is necessary to install MMCV with a version corresponding to that of PyTorch and CUDA. 

需要安装和torch、cuda对应版本的：[**mmcv**](https://mmcv.readthedocs.io/en/latest/get_started/installation.html )
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

## Verify environment(验证环境)
download config and checkpoints in：
``` 
cd .\mmsegmentation-1.x\
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
``` 
you can see config(**pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py**) and (**pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth**) in the current working directory 
</br>你可以在当前工作目录看到 **pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py** | **pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth**

test
``` 
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```
You will see a new image named result.jpg in the current folder.</br>
你将在当前文件夹中看到一个新图像 result.jpg
<p align="center">
  <img src="./imgs/pspnet.png" style="width:90%;">
</p>


