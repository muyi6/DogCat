import os

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class DogCat(Dataset):
    def __init__(self, root, transforams=None, train=True, test=False):
        self.test = test
        imgs = [os.path.join(root, file) for file in os.listdir(root)]
        # train: data/train/cat.10004.jpg
        # test1: data/test1/8973.jpg
        if test:
            imgs = sorted(imgs, key=lambda  x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda  x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        if test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(imgs_num*0.7)]
        else:
            self.imgs = imgs[int(imgs_num*0.7):]

        if transforams is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            self.transforms = T.Compose([
                T.Scale(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Scale(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])


    def __getitem__(self, index):
        assert  index >= len(self.imgs), 'Index out of range'
        img_path = self.imgs(index)
        if self.test:
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        image = Image.open(img_path)
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.imgs)