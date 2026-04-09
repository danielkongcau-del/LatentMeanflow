import os
import cv2
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import enet_weighing, median_freq_balancing
import torch.nn as nn
from collections import OrderedDict, Counter
import random
from utils import add_mask_to_source_multi_classes, add_mask_to_source


def get_class_weights(loader, out_channels, weighting):
    print('Weighting method is:{}, please wait.'.format(weighting))
    if weighting == 'enet':
        class_weights = enet_weighing(loader, out_channels)
        class_weights = torch.from_numpy(class_weights).float().cuda()
    elif weighting == 'mfb':
        class_weights = median_freq_balancing(loader, out_channels)
        class_weights = torch.from_numpy(class_weights).float().cuda()
    else:
        class_weights = None
    return class_weights


class PILToLongTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))
        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()
        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()


class SegDataset(Dataset):
    def __init__(self, dataset_dir, num_classes=2, appoint_size=(512, 512), erode=0, aug=False):
        self.imgs_dir = os.path.join(dataset_dir, 'images')
        self.labels_dir = os.path.join(dataset_dir, 'masks')
        if not os.path.isdir(self.imgs_dir) and not os.path.isdir(self.labels_dir):
            alt_imgs_dir = os.path.join(dataset_dir, 'image')
            alt_labels_dir = os.path.join(dataset_dir, 'mask')
            if os.path.isdir(alt_imgs_dir) and os.path.isdir(alt_labels_dir):
                self.imgs_dir = alt_imgs_dir
                self.labels_dir = alt_labels_dir

        if not os.path.isdir(self.imgs_dir) or not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(
                f"Dataset folders not found. Expected 'images'/'masks' or 'image'/'mask' under: {dataset_dir}"
            )

        label_files = [f for f in os.listdir(self.labels_dir)
                       if os.path.isfile(os.path.join(self.labels_dir, f))]

        image_files = [f for f in os.listdir(self.imgs_dir)
                       if os.path.isfile(os.path.join(self.imgs_dir, f))]
        image_map = {}
        for fname in image_files:
            stem, _ = os.path.splitext(fname)
            if stem not in image_map:
                image_map[stem] = fname

        samples = []
        missing_images = []
        for label_name in label_files:
            stem, _ = os.path.splitext(label_name)
            image_name = image_map.get(stem)
            if image_name is None:
                missing_images.append(label_name)
                continue
            label_path = os.path.join(self.labels_dir, label_name)
            img_path = os.path.join(self.imgs_dir, image_name)
            samples.append((img_path, label_path))

        if missing_images:
            missing_preview = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"Missing images for {len(missing_images)} masks. Example(s): {missing_preview}"
            )

        if not samples:
            raise FileNotFoundError(
                f"No valid image/mask pairs found under: {dataset_dir}"
            )

        self.samples = samples
        self.num_classes = num_classes
        self.appoint_size = appoint_size
        self.erode = 0 if erode is None else erode
        self.aug = aug

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, i):
    #     name = self.names[i]
    #     label_path = os.path.join(self.labels_dir, name)
    #     img_path = os.path.join(self.imgs_dir, name[:-3] + 'jpg')  # jpg
    #     image = cv2.imread(img_path)
    #     label = Image.open(label_path).convert('L')
    #     label = np.array(label)
    #     label[label > 0] = 1.0
    #     label = Image.fromarray(label)
    #     if self.aug:
    #         random_down_factor = random.uniform(1, 5)
    #         new_size = (int(2448 // random_down_factor), int(2048 // random_down_factor))
    #         image = cv2.resize(image, new_size)
    #         # image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    #         label = label.resize((new_size[1], new_size[0]), Image.NEAREST)
    #
    #     img_transform = transforms.Compose(
    #         [transforms.ToPILImage(), transforms.Resize(self.appoint_size), transforms.ToTensor()])
    #     img_tensor = img_transform(image)
    #
    #     label = label.resize((self.appoint_size[1], self.appoint_size[0]), Image.NEAREST)
    #     if self.erode > 0:
    #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode, self.erode))
    #         label_np = cv2.erode(np.array(label), kernel)
    #         label = Image.fromarray(label_np)
    #     if self.num_classes == 1:
    #         label_transform = transforms.Compose(transforms.ToTensor())
    #     else:
    #         label_transform = transforms.Compose([PILToLongTensor()])
    #
    #     label_tensor = label_transform(label)
    #
    #     return img_tensor, label_tensor
    
    # # # CV2 
    # def __getitem__(self, i):
    #     name = self.names[i]
    #     label_path = os.path.join(self.labels_dir, name)
    #     img_path = os.path.join(self.imgs_dir, name[:-3] + 'jpg')  # jpg or png

    #     image = cv2.imread(img_path)
    #     label = Image.open(label_path).convert('L')
    #     if self.aug:
    #         random_down_factor = random.uniform(1, 5)
    #         new_size = (int(2448 // random_down_factor), int(2048 // random_down_factor))
    #         image = cv2.resize(image, new_size)
    #         label = label.resize((new_size[1], new_size[0]), Image.NEAREST)

    #     img_transform = transforms.Compose(
    #         [transforms.ToPILImage(), transforms.Resize(self.appoint_size), transforms.ToTensor()])
    #     img_tensor = img_transform(image)

    #     label = label.resize((self.appoint_size[1], self.appoint_size[0]), Image.NEAREST)
    #     if self.erode > 0:
    #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode, self.erode))
    #         label_np = cv2.erode(np.array(label), kernel)
    #         label = Image.fromarray(label_np)
    #     if self.num_classes == 1:
    #         label_transform = transforms.Compose(transforms.ToTensor())
    #     else:
    #         label_transform = transforms.Compose([PILToLongTensor()])

    #     label_tensor = label_transform(label)

    #     return img_tensor, label_tensor

    def __getitem__(self, i):
        img_path, label_path = self.samples[i]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.aug:
            random_down_factor = random.uniform(1, 5)
            new_size = (int(2448 // random_down_factor), int(2048 // random_down_factor))
            image = image.resize(new_size, Image.BILINEAR)
            label = label.resize(new_size, Image.NEAREST)

        img_transform = transforms.Compose(
            [transforms.Resize(self.appoint_size), transforms.ToTensor()])
        img_tensor = img_transform(image)

        label = label.resize((self.appoint_size[1], self.appoint_size[0]), Image.NEAREST)
        if self.erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode, self.erode))
            label_np = cv2.erode(np.array(label), kernel)
            label = Image.fromarray(label_np)
        if self.num_classes == 1:
            label_tensor = transforms.ToTensor()(label)
        else:
            label_tensor = transforms.Compose([PILToLongTensor()])(label)

        return img_tensor, label_tensor


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    aug = False
    num_classes = 4
    appoint_size = (512, 512)
    dataset_dir = 'Data/aug/val'

    dataset = SegDataset(dataset_dir, num_classes=num_classes, appoint_size=appoint_size, erode=0, aug=aug)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, batch_data in enumerate(loader):
        if i % 100 == 0:
            print('Check done', i)
