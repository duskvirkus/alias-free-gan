from torch.utils import data
# from src.stylegan2.dataset import MultiResolutionDataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FakeMultiResolutionDataset(Dataset):
    def __init__(self, resolution=256):
        self.resolution = resolution

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # with self.env.begin(write=False) as txn:
        #     key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
        #     img_bytes = txn.get(key)

        # buffer = BytesIO(img_bytes)
        # img = Image.open(buffer)
        # img = self.transform(img)

        # return img
        a = np.random.rand(self.resolution, self.resolution, 3) * 255
        img = Image.fromarray(a.astype('uint8')).convert('RGB')
        return img


def get_fake_dataloader(size):

    dataset = FakeMultiResolutionDataset(resolution=size)

    train_loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)

    return train_loader
