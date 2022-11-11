import torch
import torchvision.transforms as transforms
from utils.base_dataset import Normalize_image
from PIL import Image
import random

class pedestrain_dataset(torch.utils.data.Dataset):
    def __init__(self, img_list, show_mode=False, max_images=None):
        " This dataset is for inference only"

        super(pedestrain_dataset,self).__init__()
        # exclude any file that is not an image:
        valid = [".jpg", ".jpeg", ".png"]
        self.img_list = [i for i in img_list if any(j in i for j in valid)]
        if max_images:
            random.seed(10)
            print("Shuffling images and taking only " + str(max_images) + " samples")
            random.shuffle(self.img_list)
            self.img_list = self.img_list[0:max_images]
        self.show_mode = show_mode

        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(0.5, 0.5)]
        self.transform = transforms.Compose(transforms_list)

    def __getitem__(self,idx):
        img = self.img_list[idx]
        img = Image.open(img).convert("RGB")
        img_size = img.size
        img_resize = img.resize((64,128), resample = Image.BICUBIC)
        if self.show_mode:
            return self.transform(img_resize), img_size
        else:
            return self.transform(img_resize), []

    def __len__(self):
        return len(self.img_list)