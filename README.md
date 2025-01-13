# MyGS_Learning

# 环境配置
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Pytorch3D 0.7.5
pip install --no-index --no-cache-dir pytorch3d -f https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu118_pyt210.tar.bz2

conda install /media/ailab_306/pro990/packages/linux-64_pytorch3d-0.7.5-py39_cu118_pyt201.tar.bz2

conda install -c fvcore -c iopath -c conda-forge fvcore iopath

在使用Hugging Face的模型时，通常使用的是from_pretrained方法来加载模型。默认情况下，这个方法会先查询网络最新版本 (通常报错在这里)，再检查本地是否已有该模型，如果没有，则会从Hugging Face的模型库中下载。

如果你想优先使用本地已有的模型，并且只在缺少的情况下从网络上下载，你可以使用以下方法：

1 使用from_pretrained方法： 这个方法是检测本地是否有模型的，如果没有，它将自动下载。所以，如果你只是想确保始终使用最新的模型，你可以定期手动更新本地模型，或者使用版本控制来跟踪模型的更新。

```python

from transformers import BertModel
model = BertModel.from_pretrained('bert-base-uncased')
```



2 手动检查模型文件： 如果想要自己控制下载逻辑，可以先检查本地是否有该模型，如果没有，再使用from_pretrained。

```python

from transformers import BertModel
import os
model_name = 'bert-base-uncased'
local_model_path = f'./{model_name}'
if os.path.exists(local_model_path):
    model = BertModel.from_pretrained(local_model_path)
else:
    model = BertModel.from_pretrained(model_name)

```

3 使用环境变量： 你还可以通过设置环境变量TRANSFORMERS_OFFLINE为1来强制transformers库使用离线模式。在这个模式下，from_pretrained将只尝试从本地加载模型，如果找不到，则会抛出异常。

```python

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import BertModel
try:
    model = BertModel.from_pretrained('bert-base-uncased')
except Exception as e:
    print("无法从本地加载模型，请检查模型文件或网络连接。")
```



4 使用镜像 export HF_ENDPOINT=https://hf-mirror.com  并donate。 作者：带娃爱好者 https://www.bilibili.com/read/cv32290878/ 出处：bilibili



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

v39:
生成器学习率：0.0006
判别器学习率：0.0001
GAN：0.01
![image](https://github.com/user-attachments/assets/9ef59b0e-f393-46db-b321-c50a5805efb2)

v40:
生成器学习率：0.001
判别器学习率：0.0001
GAN：0.01

v41:
生成器学习率：0.0005
判别器学习率：0.0001
GAN：0.01

v42：
生成器学习率：0.0005
判别器学习率：0.001
GAN：0.01

v44：
生成器学习率：0.0005
判别器学习率：0.0005
GAN：0.001

v45：
生成器学习率：0.0005
判别器学习率：0.01
GAN：0.001
