import os
import json
import cv2
import numpy as np
from PIL import Image
import base64
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import albumentations as A

class DefectDataProcessor:
    def __init__(self, data_dir, output_dir, img_size=(512, 512)):
        """
        初始化数据处理器
        
        Args:
            data_dir: 原始数据目录
            output_dir: 处理后数据输出目录
            img_size: 调整后的图像大小
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "train", exist_ok=True)
        os.makedirs(self.output_dir / "val", exist_ok=True)
        os.makedirs(self.output_dir / "test", exist_ok=True)
        
        # 数据增强变换
        self.transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Resize(img_size[0], img_size[1])
        ])
        
    def collect_and_organize_data(self, defect_dir="defect", normal_dir="normal"):
        """收集并组织数据"""
        defect_path = self.data_dir / defect_dir
        normal_path = self.data_dir / normal_dir
        
        defect_images = list((defect_path).glob("*.jpg")) + list((defect_path).glob("*.png"))
        normal_images = list((normal_path).glob("*.jpg")) + list((normal_path).glob("*.png"))
        
        print(f"找到 {len(defect_images)} 张缺陷图像和 {len(normal_images)} 张正常图像")
        
        # 划分数据集
        defect_train, defect_temp = train_test_split(defect_images, test_size=0.3, random_state=42)
        defect_val, defect_test = train_test_split(defect_temp, test_size=0.5, random_state=42)
        
        normal_train, normal_temp = train_test_split(normal_images, test_size=0.3, random_state=42)
        normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=42)
        
        # 处理并保存数据
        self._process_image_set(defect_train, "train", "defect")
        self._process_image_set(defect_val, "val", "defect")
        self._process_image_set(defect_test, "test", "defect")
        
        self._process_image_set(normal_train, "train", "normal")
        self._process_image_set(normal_val, "val", "normal")
        self._process_image_set(normal_test, "test", "normal")
        
        # 创建训练数据JSON
        self._create_training_json()
        
    def _process_image_set(self, image_paths, split, label):
        """处理图像集合并保存到指定目录"""
        target_dir = self.output_dir / split / label
        os.makedirs(target_dir, exist_ok=True)
        
        for i, img_path in enumerate(image_paths):
            # 读取图像
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 应用变换（仅对训练集进行数据增强）
            if split == "train":
                transformed = self.transform(image=img)
                img = transformed["image"]
            else:
                img = cv2.resize(img, self.img_size)
            
            # 保存处理后的图像
            output_path = target_dir / f"{i}_{img_path.name}"
            Image.fromarray(img).save(output_path)
    
    def _create_training_json(self):
        """创建适用于Ollama微调的训练数据JSON"""
        train_data = []
        
        # 处理缺陷图像
        defect_images = list((self.output_dir / "train" / "defect").glob("*.jpg")) + \
                        list((self.output_dir / "train" / "defect").glob("*.png"))
        
        for img_path in defect_images:
            # 读取图像并转换为base64
            with open(img_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # 创建训练样本
            sample = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "source": {"data": f"data:image/jpeg;base64,{img_base64}"}}
                    ]},
                    {"role": "assistant", "content": "这张图像显示了表面缺陷。我检测到表面有损坏痕迹，可能是划痕、凹陷或裂缝。这种缺陷位于图像中心区域，属于中等严重程度。此类缺陷可能由生产过程中的机械摩擦或冲击造成，建议进行局部修复或更换部件。"}
                ]
            }
            train_data.append(sample)
        
        # 处理正常图像
        normal_images = list((self.output_dir / "train" / "normal").glob("*.jpg")) + \
                        list((self.output_dir / "train" / "normal").glob("*.png"))
        
        for img_path in normal_images:
            with open(img_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            sample = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "source": {"data": f"data:image/jpeg;base64,{img_base64}"}}
                    ]},
                    {"role": "assistant", "content": "这张图像显示的表面没有缺陷。表面状况良好，没有发现划痕、凹陷、裂缝或其他任何形式的损伤。材料表面平整光滑，符合质量标准。"}
                ]
            }
            train_data.append(sample)
        
        # 保存训练数据JSON
        with open(self.output_dir / "training_data.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
            
        print(f"创建了包含 {len(train_data)} 个训练样本的JSON文件")

if __name__ == "__main__":
    # 示例用法
    processor = DefectDataProcessor(
        data_dir="./raw_data",
        output_dir="./processed_data"
    )
    processor.collect_and_organize_data()
