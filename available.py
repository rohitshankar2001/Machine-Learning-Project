from torchvision import datasets, transforms
from data.manipulate import UnNormalize, permutate_image_pixels
import torch
import numpy as np
from imagecorruptions import corrupt, get_corruption_names
import matplotlib.pyplot as plt

class corruption():
    def __init__(self, corruption_name, severity):
        self.corruption_name = corruption_name
        self.severity = severity

    def __call__(self, img):
        # print("RUNNING MNIST")
        img_np = np.array(img)
        img_corrupted = corrupt(img_np,self.severity,self.corruption_name)
        # plt.imshow(img_corrupted)
        # plt.savefig("testGaus2")
        return img_corrupted
    
class subtractLabel:
    def __init__(self, label):
        self.label = label
    def __call__(self,target):
        transformedTarget = target - self.label
        return transformedTarget

class multContextClass:
    def __init__(self, context_id, classes_per_context):
        self.context_id = context_id
        self.classes_per_context = classes_per_context
    def __call__(self,target):
        transformedTarget = target + self.context_id*self.classes_per_context
        return transformedTarget

class permute_img:
    def __init__(self, perm):
        self.perm = perm
    def __call__(self,target):
        transformedTarget = permutate_image_pixels(target, self.perm) 
        return transformedTarget    

# specify available data-sets.
AVAILABLE_DATASETS = {
    'MNIST': datasets.MNIST,
    'CIFAR100': datasets.CIFAR100,
    'CIFAR10': datasets.CIFAR10,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'MNIST': [
        transforms.ToTensor(),
    ],
    'MNIST32': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'CIFAR10': [
        transforms.ToTensor(),
    ],
    'CIFAR100': [
        transforms.ToTensor(),
    ],
    'CIFAR10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    # 'CIFAR10_corruption': [
    #     corruption(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    # ],
    # 'CIFAR100_corruption': [
    #     #corruption(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])

    # ],
    'CIFAR100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'CIFAR10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'CIFAR100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'MNIST': {'size': 28, 'channels': 1, 'classes': 10},
    'MNIST32': {'size': 32, 'channels': 1, 'classes': 10},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
    'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
    'ImageNet': {'size': 224, 'channels':3, 'classes': 50}
}
