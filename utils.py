<<<<<<< HEAD
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Grayscale, RandomHorizontalFlip, RandomRotation
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
import random
import cv2
from config import config


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class AddBlur(object):
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class Custom_Dataset(Dataset):
    def __init__(self, img_root, mask_root, video_root, img_transform, video_transform, csv_path, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.img_root = img_root
        self.mask_root = mask_root
        self.video_root = video_root
        self.img_transform = img_transform
        self.video_transform = video_transform
        self.csv = csv_path
        df = pd.read_csv(self.csv)
        self.info = df

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        file_name = patience_info['name']
        file_path = glob(self.video_root+'/'+file_name)[0]
        file_name = file_name.split('.')[0]
        label = patience_info['label']
        cap = cv2.VideoCapture(file_path)

        # 获取视频的总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算要等间隔选择的帧数
        step = total_frames // self.num_frames

        frames = []
        for frame_idx in range(step, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                if self.video_transform:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(np.uint8(frame))
                    frame = self.video_transform(frame)
                frames.append(frame)

        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        # 关闭视频
        cap.release()

        # 将帧堆叠成一个3D张量
        frames = np.stack(frames)

        # 将帧转换为PyTorch张量
        frames = torch.from_numpy(frames).permute(1, 0, 2, 3).float()

        image_path = self.img_root + '/' + file_name + '.png'
        image = Image.open(image_path)
        mask_path = self.mask_root + '/' + file_name + '.png'
        mask = Image.open(mask_path)
        mask_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        mask = mask_transform(mask)
        if self.img_transform:
            image = self.img_transform(image)

        return {'images': image, 'videos': frames, 'labels': label, 'masks': mask, 'names': file_name}

    def __len__(self):
        return len(self.info)

    def get_labels(self):                           #  添加的用于获取标签的代码
        return self.info['label'].values

def get_dataset(imgpath, maskpath, videopath, csvpath, img_size, mode='train', keyword=None):
    video_train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.CenterCrop((img_size, img_size)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        # transforms.RandomRotation((-20, 20)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    video_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    img_train_transform = transforms.Compose([
        transforms.Resize((img_size * 2, img_size * 2)),
        transforms.Grayscale(),
        transforms.CenterCrop((img_size * 2, img_size * 2)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        # transforms.RandomRotation((-20, 20)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    img_test_transform = transforms.Compose([
        transforms.Resize((img_size * 2, img_size * 2)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    if mode =='train':
        img_transform = img_train_transform
        video_transform = video_train_transform
    elif mode == 'test':
        img_transform = img_test_transform
        video_transform = video_test_transform

    dataset = Custom_Dataset(imgpath, maskpath, videopath, img_transform, video_transform, csvpath, num_frames=16)

    return dataset

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix


# # exemple
# train_set = get_dataset(imgpath='/data_chi/wubo/data/US/img', maskpath='/data_chi/wubo/data/ROI',
#                             videopath='/data_chi/wubo/data/CEUS/video', csvpath='/data_chi/wubo/data/CEUS/TNUS.csv',
#                             img_size=224)
# print(train_set[0])
=======
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Grayscale, RandomHorizontalFlip, RandomRotation
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
import random
import cv2
from config import config


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class AddBlur(object):
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class Custom_Dataset(Dataset):
    def __init__(self, img_root, mask_root, video_root, img_transform, video_transform, csv_path, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.img_root = img_root
        self.mask_root = mask_root
        self.video_root = video_root
        self.img_transform = img_transform
        self.video_transform = video_transform
        self.csv = csv_path
        df = pd.read_csv(self.csv)
        self.info = df

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        file_name = patience_info['name']
        file_path = glob(self.video_root+'/'+file_name)[0]
        file_name = file_name.split('.')[0]
        label = patience_info['label']
        cap = cv2.VideoCapture(file_path)

        # 获取视频的总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算要等间隔选择的帧数
        step = total_frames // self.num_frames

        frames = []
        for frame_idx in range(step, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                if self.video_transform:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(np.uint8(frame))
                    frame = self.video_transform(frame)
                frames.append(frame)

        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        # 关闭视频
        cap.release()

        # 将帧堆叠成一个3D张量
        frames = np.stack(frames)

        # 将帧转换为PyTorch张量
        frames = torch.from_numpy(frames).permute(1, 0, 2, 3).float()

        image_path = self.img_root + '/' + file_name + '.png'
        image = Image.open(image_path)
        mask_path = self.mask_root + '/' + file_name + '.png'
        mask = Image.open(mask_path)
        mask_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        mask = mask_transform(mask)
        if self.img_transform:
            image = self.img_transform(image)

        return {'images': image, 'videos': frames, 'labels': label, 'masks': mask, 'names': file_name}

    def __len__(self):
        return len(self.info)

    def get_labels(self):                           #  添加的用于获取标签的代码
        return self.info['label'].values

def get_dataset(imgpath, maskpath, videopath, csvpath, img_size, mode='train', keyword=None):
    video_train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.CenterCrop((img_size, img_size)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        # transforms.RandomRotation((-20, 20)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    video_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    img_train_transform = transforms.Compose([
        transforms.Resize((img_size * 2, img_size * 2)),
        transforms.Grayscale(),
        transforms.CenterCrop((img_size * 2, img_size * 2)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        # transforms.RandomRotation((-20, 20)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    img_test_transform = transforms.Compose([
        transforms.Resize((img_size * 2, img_size * 2)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    if mode =='train':
        img_transform = img_train_transform
        video_transform = video_train_transform
    elif mode == 'test':
        img_transform = img_test_transform
        video_transform = video_test_transform

    dataset = Custom_Dataset(imgpath, maskpath, videopath, img_transform, video_transform, csvpath, num_frames=16)

    return dataset

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix


# # exemple
# train_set = get_dataset(imgpath='/data_chi/wubo/data/US/img', maskpath='/data_chi/wubo/data/ROI',
#                             videopath='/data_chi/wubo/data/CEUS/video', csvpath='/data_chi/wubo/data/CEUS/TNUS.csv',
#                             img_size=224)
# print(train_set[0])
>>>>>>> 661c694 ('init')
