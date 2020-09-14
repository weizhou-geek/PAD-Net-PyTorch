import os
import torch
import random
import cv2
import scipy.io as scio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.utils.data as data
from torchvision import transforms
import datasets.transforms as mytransforms


def get_dataset(is_training):
    img_path_list = []
    img_name_list = []
    datasets_list = []
    sets_path = get_setspath(is_training)
    print(sets_path)
    labels_path = get_labelspath(is_training)
    transform = get_transform()
    for set_path in sets_path:
        subset_names = os.listdir(set_path)
        for subset_name in subset_names:
            subset_path = os.path.join(set_path, subset_name)
            img_name_list.append(subset_name)
            img_path_list.append(subset_path)

    datasets_list.append(
        ImgDataset(
            img_path=img_path_list,
            img_name=img_name_list,
            transform=transform,
            is_training=is_training,
            label_path=labels_path[0]
        )
    )
    return data.ConcatDataset(datasets_list)



def get_setspath(is_training):
    sets_root = './database/'
    if is_training:
        sets = [
            'trainimg_live1'
        ]
    else:
        sets = [
            'testimg_live1'
        ]
    return [os.path.join(sets_root, set) for set in sets]

def get_labelspath(is_training):
    sets_root = './database/'
    if is_training:
        sets = [
            'trainlabel_live1'
        ]
    else:
        sets = [
            'testlabel_live1'
        ]
    return [os.path.join(sets_root, set) for set in sets]


def get_transform():
    # if is_training:
    #     return transforms.Compose([
    #         transforms.RandomCrop(256),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip(),
    #         transforms.ToTensor()
    #     ])
    # else:
    #     return transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    return transforms.Compose([
        transforms.ToTensor()
    ])


class ImgDataset(data.Dataset):
    def __init__(self, img_path, img_name, transform, is_training, label_path, shuffle=False):
        self.img_path = img_path
        self.img_name = img_name
        self.nSamples = len(self.img_path)
        self.indices = range(self.nSamples)
        if shuffle:
            random.shuffle(self.indices)
        self.transform = transform
        self.is_training = is_training
        self.label_path = label_path

    def __getitem__(self, index):
        img = self.img_path[index]
        imagename = self.img_name[index]
        labelname = imagename[:-4]+'.mat'
        label = os.path.join(self.label_path, labelname)
        label = scio.loadmat(label)
        label = label['score']
        label = label[0]

        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        imgl = img[:, 0:w // 2, ...]
        imgr = img[:, w // 2:w, ...]

        img_group = []
        img_group.append(imgl)
        img_group.append(imgr)

        if self.is_training:
            img_group = mytransforms.group_random_crop(img_group, [256, 256])
            img_group = mytransforms.group_random_fliplr(img_group)
            img_group = mytransforms.group_random_flipud(img_group)
            imgl = self.transform(img_group[0].copy())
            imgr = self.transform(img_group[1].copy())
            label = torch.from_numpy(label).float()
        else:
            imgl = self.transform(img_group[0])
            imgr = self.transform(img_group[1])
            label = torch.from_numpy(label).float()
        return imgl, imgr, label, imagename[:-4]

    def __len__(self):
        return self.nSamples

