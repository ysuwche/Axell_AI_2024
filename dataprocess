import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import albumentations as A
from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import PIL
import torch
from torch import Tensor
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
from IPython.display import display
from tqdm import tqdm, trange

class DataSetBase(data.Dataset, ABC):
    def __init__(self, image_path: Path):
        self.images = list(image_path.iterdir())
        self.max_num_sample = len(self.images)

    def __len__(self) -> int:
        return self.max_num_sample

    @abstractmethod
    def get_low_resolution_image(self, image: Image, path: Path)-> Image:
        pass

    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return image

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        image_path = self.images[index % len(self.images)]
        high_resolution_image = self.preprocess_high_resolution_image(PIL.Image.open(image_path))
        low_resolution_image = self.get_low_resolution_image(high_resolution_image, image_path)
        return transforms.ToTensor()(low_resolution_image), transforms.ToTensor()(high_resolution_image)

class TrainDataSet(DataSetBase):
    def __init__(self, image_path: Path, num_image_per_epoch: int = 2000):
        super().__init__(image_path)
        self.max_num_sample = num_image_per_epoch

    def get_low_resolution_image(self, image: Image, path: Path)-> Image:
        return transforms.Resize((image.size[0] // 4, image.size[1] // 4), transforms.InterpolationMode.BICUBIC)(image.copy())

    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return transforms.Compose([
            transforms.RandomCrop(size = 512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.5)
        ])(image.copy())

class TrainDataSetNoAug(DataSetBase):
    def __init__(self, image_path: Path, num_image_per_epoch: int = 2000):
        super().__init__(image_path)
        self.max_num_sample = num_image_per_epoch

    def get_low_resolution_image(self, image: Image, path: Path)-> Image:
        return transforms.Resize((image.size[0] // 4, image.size[1] // 4), transforms.InterpolationMode.BICUBIC)(image.copy())

    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return transforms.Compose([
            transforms.RandomCrop(size = 512),
        ])(image.copy())

class ValidationDataSet(DataSetBase):
    def __init__(self, high_resolution_image_path: Path, low_resolution_image_path: Path):
        super().__init__(high_resolution_image_path)
        self.high_resolution_image_path = high_resolution_image_path
        self.low_resolution_image_path = low_resolution_image_path

    def get_low_resolution_image(self, image: Image, path: Path)-> Image:
        return PIL.Image.open(self.low_resolution_image_path / path.relative_to(self.high_resolution_image_path))

def get_dataset() -> Tuple[TrainDataSet, TrainDataSetNoAug, ValidationDataSet]:
    return TrainDataSet(Path("./dataset/train/train"), 850 * 10), TrainDataSetNoAug(Path("./dataset/train/train"), 850 * 10), ValidationDataSet(Path("./dataset/validation/validation/original"), Path("./dataset/validation/validation/0.25x"))
