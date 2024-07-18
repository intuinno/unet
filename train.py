import torch
from torch import nn
import numpy as np
import random
import os
import pathlib
from carvana import CarvanaDataset
from unet import UNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
from types import SimpleNamespace
from torchvision.transforms.functional import center_crop

config = {
    "seed": 42,
    "device": "cpu",
    "data_dir": "/Users/intuinno/codegit/dataset/carvana",
    "mask_channels": 1,
    "image_channels": 3,
    "batch_size": 128,
    "learning_rate": 2.5e-4,
    "epochs": 1000000,
    "eval_every": 10,
}

config = SimpleNamespace(**config)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # enable deterministic run
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def main():
    set_seed_everywhere(config.seed)
    data_dir = pathlib.Path(config.data_dir)
    dataset = CarvanaDataset(data_dir / "train", data_dir / "train_masks")
    model = UNet(config.image_channels, config.mask_channels).to(config.device)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        config.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logger = SummaryWriter()
    sigmoid = nn.Sigmoid()
    loss_func = nn.BCELoss()

    for epoch in tqdm(range(config.epochs)):
        # Evaluate model
        if epoch % config.eval_every == 0:
            with torch.no_grad():
                x, _ = next(iter(data_loader))
                x = x.to(config.device)
                mask = sigmoid(model(x))
                x = center_crop(x, [mask.shape[2], mask.shape[3]])
                x = x[:9]
                grid = torchvision.utils.make_grid(x)
                logger.add_image("images", grid, epoch)
                logger.flush()

        # Train model
        image, mask = next(iter(data_loader))
        image, mask = image.to(config.device), mask.to(config.device)
        logits = model(image)
        mask = center_crop(mask, [logits.shape[2], logits.shape[3]])
        loss = loss_func(sigmoid(logits), mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % config.eval_every == 0:
            logger.add_scalar("training_loss", loss, epoch)
            print(f"Step: {epoch},\tLoss: {loss}")


if __name__ == "__main__":
    main()
