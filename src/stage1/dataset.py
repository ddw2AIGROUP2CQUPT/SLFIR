import numpy as np
import torch
from glob import glob
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SLFIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode

        if hp.dataset_name == "Face-1000":
            self.root_dir = os.path.join(hp.root_dir, 'Dataset', '1000')
        elif hp.dataset_name == "Face-450":
            self.root_dir = os.path.join(hp.root_dir, 'Dataset', '450')
        # Windows环境更改为以下路径
        # self.root_dir = os.path.join(r"C:\Users\wrtan\Desktop\tangger\PycharmProjects\face-sbir", 'Dataset')
        self.train_photo_paths = sorted(glob(os.path.join(self.root_dir, 'comp', 'train','photo', '*')))
        self.train_sketch_paths = sorted(glob(os.path.join(self.root_dir, 'comp', 'train', 'sketch', '*')))
        self.test_photo_paths = sorted(glob(os.path.join(self.root_dir, 'comp', 'test', 'photo', '*')))
        self.test_sketch_paths = sorted(glob(os.path.join(self.root_dir, 'comp', 'test', 'sketch', '*')))

        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')

    def __getitem__(self, item):
        sample = {}
        if self.mode == 'Train':
            sketch_path = self.train_sketch_paths[item]
            positive_path = self.train_photo_paths[item]
            negative_path = self.train_photo_paths[randint(0, len(self.train_photo_paths) - 1)]

            # 注意：这里要对图像取反，因为原始图像是白底黑字，而训练集中要求的图像是黑底白字
            sketch_img = 255 - np.array(Image.open(sketch_path).convert('RGB'))
            sketch_img = Image.fromarray(sketch_img).convert('RGB')
            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch_img': sketch_img, 'positive_img': positive_img, 'negative_img': negative_img, 
                        'sketch_path': sketch_path, 'positive_path': positive_path, 'negative_path': negative_path}

        elif self.mode == 'Test':
            
            sketch_path = self.test_sketch_paths[item]
            positive_path = self.test_photo_paths[item]

            # 注意：这里要对图像取反，因为原始图像是白底黑字，而训练集中要求的图像是黑底白字
            sketch_img = 255 - np.array(Image.open(sketch_path).convert('RGB'))
            sketch_img = Image.fromarray(sketch_img).convert('RGB')
            positive_img = Image.open(positive_path).convert('RGB')

            sketch_img = self.test_transform(sketch_img)
            positive_img = self.test_transform(positive_img)

            sample = {'sketch_img': sketch_img, 'positive_img': positive_img, 
                        'sketch_path': sketch_path, 'positive_path': positive_path}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.train_photo_paths)
        elif self.mode == 'Test':
            return len(self.test_photo_paths)


def get_dataloader(hp):
    dataset_Train = SLFIR_Dataset(hp, mode='Train')
    # 返回的是每一分个分支train集
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True, num_workers=int(hp.nThreads))

    dataset_Test = SLFIR_Dataset(hp, mode='Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False, num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test


def get_transform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(320), transforms.RandomCrop(299)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)
