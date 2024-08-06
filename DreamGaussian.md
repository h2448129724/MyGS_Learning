# step1

python process.py data/dog.png 

这段代码的主要作用是从给定路径中加载图像，移除背景，并对图像进行裁剪和居中处理。具体步骤如下：

1. **导入库**

2. **定义BLIP2类**：
    - **初始化**：在初始化方法中，加载了一个用于生成文本的预训练模型`Salesforce/blip2-opt-2.7b`。
    - **调用**：将输入的图像转为PIL格式，然后使用预处理器将图像转换为张量，并使用模型生成文本。

3. **主程序**：
    - 使用`argparse`解析命令行参数，包括图像路径、背景移除模型、输出分辨率、边框比例和是否重新居中。
    - 创建一个rembg会话。
    - 检查输入路径是否为目录，如果是目录则处理目录中的所有文件，否则处理单个文件。
    - 遍历文件列表，对每个文件：
        1. **加载图像**：使用`cv2.imread`加载图像。
        2. **移除背景**：使用`rembg`移除图像背景。
        3. **重新居中**（如果指定）：对处理后的图像进行居中和缩放处理，以适应指定的输出尺寸。
        4. **保存图像**：将处理后的图像保存为带有透明背景的PNG格式。

代码运行的具体步骤如下：

1. 从命令行获取输入图像路径和其他参数。
2. 使用rembg库移除图像背景。
3. 根据是否重新居中，对图像进行裁剪和调整大小。
4. 将处理后的图像保存到指定位置。

这段代码结合了深度学习模型和图像处理技术，可以用于从图像中提取物体并生成相关的描述文本。

Output：` name_rgba.png`



# Step 2

```bash
python main.py --config configs/image.yaml input=data/dog_rgba.png save_path=dog
```

## 代码的执行流程如下：

### 导入模块和依赖：

- 导入必要的 Python 模块和库。

### 解析命令行参数：

- 使用 `argparse` 和 `omegaconf` 解析命令行参数，获取配置文件路径 `configs/image.yaml` 以及额外参数 `input=data/anya_rgba.png` 和 `save_path=anya`。

### 加载配置文件：

- 使用 `OmegaConf` 加载 YAML 配置文件，并将命令行参数与配置文件合并，生成最终的配置对象 `opt`。

### 初始化 GUI 对象：

- 创建 GUI 类的实例，传递配置对象 `opt`。
- 在 GUI 的初始化方法 `__init__` 中，进行一系列初始化操作：
  - 初始化渲染器、相机、设备、输入图像等。
  - 如果命令行参数中提供了 `input`，则调用 `load_input` 方法加载输入图像。
  - 根据命令行参数设置 `prompt` 和 `negative_prompt`。
  - 根据命令行参数加载或初始化渲染器。

### 执行主流程：

- 根据配置文件中的 `gui` 参数决定是启动 GUI 还是直接进行训练：
  - 如果 `opt.gui` 为 `True`，则调用 `gui.render()` 方法启动 GUI 界面，进入渲染循环。
  - 如果 `opt.gui` 为 `False`，则调用 `gui.train(opt.iters)` 方法进行训练，训练 `opt.iters` 轮。

## 详细流程

### 1. GUI 类初始化 (`__init__` 方法)

- 设置相机参数、渲染模式、随机种子等。
- 初始化背景移除器、渲染器等模型和工具。
- 如果提供了输入数据，则加载输入图像和掩码。
- 如果提供了加载检查点，则初始化渲染器；否则，使用默认参数初始化。
- 如果启用了 GUI，则创建 DearPyGui 上下文和窗口控件。

### 2. 加载输入图像 (`load_input` 方法)

- 读取指定路径的图像文件，如果图像是 RGBA 格式，则使用 `rembg` 移除背景。
- 将图像调整为指定大小，并进行归一化处理。
- 读取与图像同名的文本文件（如果存在）作为 `prompt`。

### 3. 训练过程 (`train_step` 方法)

- 在训练过程中，每个步骤包括以下操作：
  - 渲染图像并计算损失，包括 RGB 损失和掩码损失。
  - 计算指导损失（如果启用 SD 或 Zero123 模型）。
  - 进行反向传播和优化。
  - 定期进行密度化和修剪操作。

### 4. 渲染过程 (`test_step` 方法)

- 渲染当前视角的图像，并将其更新到 GUI 界面。

### 5. 保存模型 (`save_model` 方法)

- 将训练后的模型保存为几何图形或带有纹理的模型文件。

输出文件：`anya_mesh.obj`.

# Step 3

```bash
python main2.py --config configs/image.yaml input=data/anya_rgba.png save_path=anya
```

## 整个代码的执行流程如下：

### 导入模块和依赖：

- 导入必要的 Python 模块和库。

### 解析命令行参数：

- 使用 `argparse` 和 `omegaconf` 解析命令行参数，获取配置文件路径 `configs/image.yaml` 以及额外参数 `input=data/anya_rgba.png` 和 `save_path=anya`。

### 加载配置文件：

- 使用 `OmegaConf` 加载 YAML 配置文件，并将命令行参数与配置文件合并，生成最终的配置对象 `opt`。

### 初始化 GUI 对象：

- 创建 GUI 类的实例，传递配置对象 `opt`。
- 在 GUI 的初始化方法 `__init__` 中，进行一系列初始化操作：
  - 初始化渲染器、相机、设备、输入图像等。
  - 如果命令行参数中提供了 `input`，则调用 `load_input` 方法加载输入图像。
  - 根据命令行参数设置 `prompt` 和 `negative_prompt`。
  - 根据命令行参数加载或初始化渲染器。

### 执行主流程：

- 根据配置文件中的 `gui` 参数决定是启动 GUI 还是直接进行训练：
  - 如果 `opt.gui` 为 `True`，则调用 `gui.render()` 方法启动 GUI 界面，进入渲染循环。
  - 如果 `opt.gui` 为 `False`，则调用 `gui.train(opt.iters_refine)` 方法进行训练，训练 `opt.iters_refine` 轮。

## 详细流程

### 1. GUI 类初始化 (`__init__` 方法)

- 设置相机参数、渲染模式、随机种子等。
- 初始化背景移除器、渲染器等模型和工具。
- 如果提供了输入数据，则加载输入图像和掩码。
- 根据命令行参数设置 `prompt` 和 `negative_prompt`。
- 初始化渲染器。
- 如果启用了 GUI，则创建 DearPyGui 上下文和窗口控件。

### 2. 加载输入图像 (`load_input` 方法)

- 读取指定路径的图像文件，如果图像是 RGBA 格式，则使用 `rembg` 移除背景。
- 将图像调整为指定大小，并进行归一化处理。
- 读取与图像同名的文本文件（如果存在）作为 `prompt`。

### 3. 训练过程 (`train_step` 方法)

- 在训练过程中，每个步骤包括以下操作：
  - 渲染图像并计算损失，包括 RGB 损失和掩码损失。
  - 计算指导损失（如果启用 SD 或 Zero123 模型）。
  - 进行反向传播和优化。
  - 定期进行密度化和修剪操作。

### 4. 渲染过程 (`test_step` 方法)

- 渲染当前视角的图像，并将其更新到 GUI 界面。

### 5. 保存模型 (`save_model` 方法)

- 将训练后的模型保存为几何图形或带有纹理的模型文件。

输出文件：`anya.obj`.

## python -m kiui.render logs/anya.obj --save_video name.mp4 --wogui



https://github.com/ForMyCat/SparseGS/blob/master/guidance/zero123_utils.py
# zero123_utils.py

![image-20240806120510910](D:\研究生\科研\DreamGaussian.assets\image-20240806120510910.png)



### 代码功能
1. **模型初始化 (`Zero123` 类)**：
   - 初始化 CLIP 模型、VAE 模型和 U-Net 模型。
   - 设置图像编码器、VAE 和 U-Net 模型为评估模式。
   - 配置 DDIM 调度器，设置训练步数和时间步长范围。

2. **获取图像嵌入 (`get_img_embeds` 方法)**：
   - 接收输入图像并调整大小到 256x256。
   - 使用 CLIP 模型提取图像嵌入。
   - 使用 VAE 模型对图像进行编码，获取图像的潜在表示。

3. **细化图像 (`refine` 方法)**：
   - 使用 DDIM 调度器设置时间步长。
   - 根据图像的极坐标角、方位角和半径生成相机嵌入。
   - 将图像嵌入和相机嵌入拼接，并通过 CLIP 相机投影层处理。
   - 使用 U-Net 模型生成噪声预测，并根据指导比例调整噪声预测。
   - 通过 DDIM 调度器逐步细化图像的潜在表示，并解码生成最终图像。

4. **训练步骤 (`train_step` 方法)**：
   - 根据图像的极坐标角、方位角和半径生成相机嵌入。
   - 使用 U-Net 模型生成噪声预测，并根据指导比例调整噪声预测。
   - 计算损失，并进行反向传播。


以下是对每段代码的分析及其功能：

### imagedream_utils.py

#### ImageDream 类
- **__init__ 方法**:
  - 初始化类，包括加载模型、设置调度器和图像嵌入。
  - 模型通过 `build_model` 函数加载，并设置为评估模式。
  - 使用 `DDIMScheduler` 设置调度器，用于扩散模型的时间步长调度。

- **get_image_text_embeds 方法**:
  - 处理图像和文本提示，生成图像和文本的嵌入表示。
  - 图像被调整大小并转换为 PIL 图像格式，然后通过模型获取嵌入。
  - 文本提示通过 `encode_text` 方法编码为嵌入。

- **encode_text 方法**:
  - 将文本提示编码为嵌入表示。
  - 使用模型的 `get_learned_conditioning` 方法生成文本嵌入。

- **refine 方法**:
  - 细化给定的 RGB 图像。
  - 包括步骤如调整图像大小、添加噪声、根据时间步长进行扩散预测等。
  - 使用 UNet 模型预测噪声，并根据指导比例进行噪声调整。

- **decode_latents 方法**:
  - 解码潜在变量为图像。
  - 使用模型的 `decode_first_stage` 方法。

- **encode_imgs 方法**:
  - 编码图像为潜在变量。
  - 使用模型的 `encode_first_stage` 和 `get_first_stage_encoding` 方法。

#### 关键功能实现：
- 图像和文本的嵌入提取。
- 使用扩散模型进行图像生成和细化。
- 解码潜在变量为最终图像。

### mvdream_utils.py

#### MVDream 类
- **__init__ 方法**:
  - 初始化类，加载模型并设置调度器。
  - 使用 `build_model` 方法加载模型，并设置为评估模式。
  - 使用 `DDIMScheduler` 设置调度器。

- **get_text_embeds 方法**:
  - 编码文本提示为嵌入表示，重复嵌入以适应批处理大小。

- **encode_text 方法**:
  - 编码文本提示为嵌入表示。

- **refine 方法**:
  - 细化给定的 RGB 图像。
  - 包括步骤如调整图像大小、添加噪声、扩散预测、噪声调整等。

- **decode_latents 方法**:
  - 解码潜在变量为图像。

- **encode_imgs 方法**:
  - 编码图像为潜在变量。

- **prompt_to_img 方法**:
  - 将提示转换为图像。
  - 包括文本嵌入提取、图像嵌入提取、摄像头参数处理和扩散采样等步骤。

#### 关键功能实现：
- 文本嵌入提取。
- 使用扩散模型进行图像生成和细化。
- 解码潜在变量为最终图像。

### sd_utils.py

#### StableDiffusion 类
- **__init__ 方法**:
  - 初始化类，加载模型并设置调度器。
  - 使用 `StableDiffusionPipeline` 从预训练模型加载必要的组件，如 VAE、UNet 和文本编码器。
  - 使用 `DDIMScheduler` 设置调度器。

- **get_text_embeds 方法**:
  - 编码文本提示为嵌入表示。

- **encode_text 方法**:
  - 使用 `tokenizer` 和 `text_encoder` 编码文本提示为嵌入表示。

- **refine 方法**:
  - 细化给定的 RGB 图像。
  - 包括步骤如调整图像大小、添加噪声、扩散预测、噪声调整等。

- **decode_latents 方法**:
  - 解码潜在变量为图像。

- **encode_imgs 方法**:
  - 编码图像为潜在变量。

- **prompt_to_img 方法**:
  - 将提示转换为图像。
  - 包括文本嵌入提取、潜在变量生成和解码为最终图像。

#### 关键功能实现：
- 文本嵌入提取。
- 使用扩散模型进行图像生成和细化。
- 解码潜在变量为最终图像。







## 使用到的关键部分。

### 初始化

```py
if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
```



   ### prepare embeddings
        with torch.no_grad():
    
            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])
    
            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)
2. **获取图像嵌入 (`get_img_embeds` 方法)**：
   
   - 接收输入图像并调整大小到 256x256。
   - 使用 CLIP 模型提取图像嵌入。
   - 使用 VAE 模型对图像进行编码，获取图像的潜在表示。
   
   ``` python
     if self.enable_zero123:
                   loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation)
               
   ```

### LOSS

**训练步骤 (`train_step` 方法)**：

   - 根据图像的极坐标角、方位角和半径生成相机嵌入。
   - 使用 U-Net 模型生成噪声预测，并根据指导比例调整噪声预测。
   - 计算损失，并进行反向传播。      



![image-20240806120402699](D:\研究生\科研\DreamGaussian.assets\image-20240806120402699.png)





zero123plus_utils.py

``` python
from diffusers import DDIMScheduler
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import sys

sys.path.append('./')


class Zero123plus(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], model_key="sudo-ai/zero123plus-v1.1"):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        assert self.fp16, 'Only zero123plus fp16 is supported for now.'

        self.pipe = DiffusionPipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            trust_remote_code=True,
        ).to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing='trailing'
        )
        self.pipe.to(self.device)
    
        self.pipe.vision_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()
       
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = None

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor_clip(images=x_pil, return_tensors="pt").pixel_values.to(device=self.device,
                                                                                                     dtype=self.dtype)
        c = self.pipe.vision_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.vae.config.scaling_factor
        self.embeddings = [c, v]


    @torch.no_grad()
    def refine(self, pred_rgb, elevation, azimuth, radius,
               guidance_scale=5, steps=50, strength=0.8, default_elevation=0,
               ):

        batch_size = pred_rgb.shape[0]

        self.scheduler.set_timesteps(steps)

        if strength == 0:
            init_step = 0
            latents = torch.randn((1, 4, 32, 32), device=self.device, dtype=self.dtype)
        else:
            init_step = int(steps * strength)
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.vision_encoder(pred_rgb_256.to(self.dtype))
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
        cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
        # cc_emb = self.pipe.vision_encoder(cc_emb)
        cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

        vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
        vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            x_in = torch.cat([latents] * 2)
            t_in = t.view(1).to(self.device)

            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents)  # [1, 3, 256, 256]
        return imgs

    def train_step(self, pred_rgb, elevation, azimuth, radius, step_ratio=None, guidance_scale=5, as_latent=False,
                   default_elevation=0):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        batch_size = pred_rgb.shape[0]

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.vision_encoder(pred_rgb_256.to(self.dtype))

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)

            T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
            cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
            cc_emb = self.pipe.vision_encoder(cc_emb)
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

            vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')

        return loss

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        latents = latents * self.vae.config.scaling_factor

        return latents


```




























`main.py` 和 `main2.py` 有很多相似之处，但也有一些显著的区别。以下是两者之间的主要区别：

### 1. 模块和依赖

- **main2.py** 额外导入了 `trimesh` 模块用于处理 3D 网格，而 **main.py** 没有。
- **main2.py** 中有一部分代码注释掉了 `kiui.lpips` 的导入，而 **main.py** 没有。

### 2. 渲染器和相机

- **main.py** 使用 `gs_renderer.Renderer` 作为渲染器，而 **main2.py** 使用 `mesh_renderer.Renderer` 作为渲染器。
- **main.py** 初始化相机时，使用 `Renderer(sh_degree=self.opt.sh_degree)`，而 **main2.py** 在初始化渲染器时传递了 `opt` 参数。

### 3. 输入图像加载

- **main.py** 的 `load_input` 方法中，使用 `rembg` 移除背景后，图像被调整大小，并转换为浮点数格式。
- **main2.py** 的 `load_input` 方法中，移除背景后，图像同样被调整大小并转换为浮点数格式，但还有一些额外的处理，如处理通道顺序和遮罩。

### 4. 训练过程

- **main.py** 的 `train_step` 方法中，渲染和计算损失的过程更为复杂，包括处理已知视图和新视图，并计算多种损失（RGB、遮罩、指导等）。
- **main2.py** 的 `train_step` 方法中，渲染和计算损失的过程相对简化，并且增加了随机 SSAA（超采样抗锯齿）的处理。

### 5. 模型保存

- **main.py** 有更详细的模型保存方法 `save_model`，可以选择保存为几何图形或带有纹理的模型。
- **main2.py** 的 `save_model` 方法相对简单，只是导出了网格。

### 6. 配置文件和命令行参数

- **main2.py** 在解析配置文件时，有一个检查 `opt.mesh` 的步骤，并尝试从默认路径加载网格，如果没有找到则抛出错误。而 **main.py** 没有这个步骤。

### 7. 图像渲染模式

- **main.py** 支持的渲染模式包括 `image`、`depth` 和 `alpha`。
- **main2.py** 支持的渲染模式包括 `image`、`depth`、`alpha` 和 `normal`。

### 8. 其他细节

- **main2.py** 在 `train_step` 方法中增加了对多视图训练的支持（如 `mvdream` 或 `imagedream`），并在指导损失的计算中使用了随机 SSAA。
- **main.py** 中有更多关于训练步骤、损失计算和优化的详细实现。

### 总结

总体来说，**main2.py** 是 **main.py** 的一个修改版本，具有以下特点：
- 更简化的训练过程。
- 额外的 3D 网格处理功能。
- 对指导模型的懒加载和额外的训练支持。
- 支持更多的渲染模式。
- 增强的输入图像处理和加载机制。

这些修改可能是为了特定的需求或优化某些性能，具体选择使用哪个文件需要根据实际需求进行评估。







# result

| model | result                    |
| ----- | ------------------------- |
| anya  | 0.6734577491879463   0.64 |
|       |                           |
|       |                           |
|       |                           |
|       |                           |
|       |                           |
|       |                           |
|       |                           |
|       |                           |

anya zero123 500
