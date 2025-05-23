# Fine-tuningLLaVA
使用自己的数据在 Windows 上微调 LLaVA 模型

参考[官方说明文档](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md)

## 克隆官方文件夹
然后直接进入这个文件夹下，之后所有的操作都在这个文件夹下。
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

## 环境配置

创建一个python虚拟环境
```bash
conda create -n llava python=3.10 -y 
conda activate llava
```

使用指令`nvidia-smi`查看你的 CUDA 版本，版本向下兼容，12.6的可以用cu118。安装 PyTorch（根据你的 CUDA 版本）。


```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

安装要求的依赖

```
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

pip install -e . 
pip install -r requirements.txt
```

## 准备自己想要用于微调的数据

LLaVA 需要一种格式为 JSON 或 JSONL 的数据，其内容包括： 
• 图像路径 
• 对话/文本内容（多轮或单轮）

### 建立文件夹结构（在下载好的LLaVA文件夹下）
```bash
mkdir -p data/my_dataset/images 
mkdir -p data/my_dataset/annotations
```

### 准备图片
将你用于训练的图像文件放到 data/my_dataset/images 文件夹下，命名建议简单（如 001.jpg）。

### 准备标注 JSON 文件
创建 data/my_dataset/annotations/train.json，格式如下（单轮对话）：
```
data = [
{
"id": "image1",
"image": "images/image1.jpg",
"conversations": [
{
"from": "human",
"value": "Describe any defects present on the parts in this image. Focus on 'Scaled', 'Peeled', or 'Rusty' defects."
},
{
"from": "gpt",
"value": "There is a half frame in the picture, and you can see about eight parts. The leftmost part has obvious peeling and scaling on its surface. The side of the part next to it, that is, in the middle of the pile of parts, also has obvious peeling and scaling. The part at the bottom has obvious peeling and scaling on its surface. All the parts in the picture have greyish (non-reflective) surfaces. This frame of parts is judged to be NC (Non-Conformance)."
}
]
},
{
"id": "image_002",
"image": "images/image2.jpg",
"conversations": [
{
"from": "human",
"value": "Are there any 'Scaled', 'Peeled', or 'Rusty' defects on the parts shown?"
},
{
"from": "gpt",
"value": "The picture shows some parts stacked together, two of which have not been penetrated at the bottom. The parts on the left have more or less obvious peeling and scaling. The lower surface of the almost complete part in the lower right corner has obvious peeling and scaling. All the parts in the picture have greyish (non-reflective) surfaces. This frame of parts is judged to be NC (Non-Conformance)."
}
]
},
{
"id": "image_003",
"image": "images/image3.jpg",
"conversations": [
{
"from": "human",
"value": "Provide a detailed report of part defects."
},
{
"from": "gpt",
"value": "The picture shows a small half frame of parts stacked together. All the parts in the picture are Greyish (Non-reflective). The parts on the left and bottom have more or less obvious peeling and scaling. This frame of parts is judged to be NC (Non-Conformance)."
}
]
}
]
```

## 开始微调 LLaVA 模型

在执行训练命令前要先设置下载模型权重路径，否则会直接下载在C盘。这是一个暂时设置，直接在当前虚拟环境中设置就行。设置好可以用第二行指令查看一下，确认已经设置好。
```bash
set HF_HOME=D:\hf_cache
echo %HF_HOME% 
```

然后运行这个命令开始训练
```bash
python llava/train/train_mem.py --model_name_or_path liuhaotian/llava-v1.5-7b --version v1 --data_path data/my_dataset/annotations/train.json --image_folder data/my_dataset/images --vision_tower openai/clip-vit-large-patch14-336 --mm_projector_type mlp2x_gelu --image_aspect_ratio pad --tune_mm_mlp_adapter True --fp16 True --output_dir ./checkpoints/llava-mydata-v1.5-7b --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 --evaluation_strategy "no" --save_strategy "epoch" --learning_rate 2e-5 --save_total_limit 2 --logging_steps 10 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 0
```

出现报错说没有可用的flash attention，直接到llava/train/train_mem.py文件中删去train括号里的东西，再运行。

17点01分，出现了新的报错，而且VS Code卡死了。周一再尝试。
