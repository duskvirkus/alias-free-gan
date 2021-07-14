import pytorch_lightning as pl

from alias_free_gan import AliasFreeGAN
from stylegan2.dataset import MultiResolutionDataset
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils import data
# from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

def cli_main(args=None):

    parser = ArgumentParser()

    script_parser = parser.add_argument_group("Trainer Script")
    script_parser.add_argument("--size", help='Pixel dimension of model. Must be 256, 512, or 1024. Required!', type=int, required=True)
    script_parser.add_argument("--dataset_path", help='Path to dataset. Required!', type=str, required=True)
    script_parser.add_argument("--batch", help='Batch size. Will be overridden if --auto_scale_batch_size is used. (default: %(default)s)', default=16, type=int)
    # TODO add support for --auto_scale_batch_size
    script_parser.add_argument("--n_samples", help='Number of samples to generate in training process. (default: %(default)s)', default=9, type=int)

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
