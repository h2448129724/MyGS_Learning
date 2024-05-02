# MyGS_Learning

# 环境配置
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia


# Pytorch3D 0.7.5
pip install --no-index --no-cache-dir pytorch3d -f https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu118_pyt210.tar.bz2
 conda install /home/ailab_306/下载/linux-64_pytorch3d-0.7.5-py39_cu118_pyt201.tar.bz2


研究思路/想法:


大致方向: 
1. 使用Diffusion Model生成点云
2. 使用生成的点云通过Gaussian Splatting重建

问题:
1. 参考已经存在的Diffusion Model+NeRF进行三维重建，由于NeRF属于隐式重建，是否可以直接套用3D GS？
2. NeRF的隐式重建可以获得模型吗，还是只能获得某个视角下观察到的图片？
3. 如何解决NeRF/GS渲染时需要的视角/相机参数问题
4. 使用Diffusion Model还算是重建吗？


下周工作计划:
1. 解决问题1-5
2. 查看论文Diffusion Model+NeRF
