from torch import nn

import torch.utils.data
import torchvision.transforms.functional
from PIL import Image


class CarvanaDataset(torch.utils.data.Dataset):

    def __init__(self, image_path, mask_path):
        self.images = {p.stem: p for p in image_path.iterdir()}
        self.masks = {p.stem[:-5]: p for p in mask_path.iterdir()}
        self.ids = list(self.images.keys())

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(572),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = Image.open(self.images[id])
        image = self.transforms(image)
        mask = Image.open(self.masks[id])
        mask = self.transforms(mask)
        mask = mask / mask.max()
        return image, mask

    def __len__(self):
        return len(self.ids)
