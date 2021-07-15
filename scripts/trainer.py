from argparse import ArgumentParser
import sys
import os

from torch.utils import data
from torchvision import transforms

import pytorch_lightning as pl
# from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.alias_free_gan import AliasFreeGAN
from src.stylegan2.dataset import MultiResolutionDataset

def cli_main(args=None):

    parser = ArgumentParser()

    script_parser = parser.add_argument_group("Trainer Script")
    script_parser.add_argument("--dataset_path", help='Path to dataset. Required!', type=str, required=True)

    parser = AliasFreeGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(args)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    print(f'Dataset path: %s' % args.dataset_path)
    
    dataset = MultiResolutionDataset(args.dataset_path, transform=transform, resolution=args.size)
    
    print(f'Initialized {dataset.__class__.__name__} dataset with {dataset.__len__()} images')

    train_loader = data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)


    # callbacks = [
    #     TensorboardGenerativeModelImageSampler(num_samples=5),
    # ]

    trainer = pl.Trainer.from_argparse_args(args)  #, callbacks=callbacks)

    model = AliasFreeGAN(**vars(args))
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    cli_main()
