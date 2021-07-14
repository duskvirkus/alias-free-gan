import pytorch_lightning as pl

from afgan import AFGAN
# from config import GANConfig
# from tensorfn import load_arg_config
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
    # script_parser.add_argument("--iter", help='Number of iterations to train for. (default: %(default)s)', default=800000, type=int)
    script_parser.add_argument("--batch", help='Batch size. Will be overridden if --auto_scale_batch_size is used. (default: %(default)s)', default=16, type=int)
    # TODO add support for --auto_scale_batch_size
    script_parser.add_argument("--n_samples", help='Number of samples to generate in training process. (default: %(default)s)', default=9, type=int)
    # script_parser.add_argument("--start_iter", help='Start iteration counter at. Useful for resuming training from a checkpoint. (default: %(default)s)', default=0, type=int)

    parser = AFGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(args)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    print(args.dataset_path)
    
    dataset = MultiResolutionDataset(args.dataset_path, transform=transform, resolution=args.size)
    
    print(f'Initialized {dataset.__class__.__name__} dataset with {dataset.__len__()} images')

    train_loader = data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)


    # callbacks = [
    #     TensorboardGenerativeModelImageSampler(num_samples=5),
    # ]

    trainer = pl.Trainer.from_argparse_args(args)  #, callbacks=callbacks)

    model = AFGAN(**vars(args))
    trainer.fit(model, train_loader)


# dataset = MultiResolutionDataset(conf.path, transform, conf.training.size)
# data_loader = data.DataLoader(
#     dataset,
#     batch_size=conf.training.batch,
#     drop_last=True,
# )

# model = AFGAN()

# trainer = pl.Trainer(progress_bar_refresh_rate=20, weights_summary='full')
# trainer.fit(model, data_loader)


# def cli_main(args=None):
#     seed_everything(1234)

#     parser = ArgumentParser()
#     parser.add_argument("--batch_size", default=64, type=int)
#     parser.add_argument("--dataset", default="mnist", type=str, choices=["lsun", "mnist"])
#     parser.add_argument("--data_dir", default="./", type=str)
#     parser.add_argument("--image_size", default=64, type=int)
#     parser.add_argument("--num_workers", default=8, type=int)

#     script_args, _ = parser.parse_known_args(args)

#     if script_args.dataset == "lsun":
#         transforms = transform_lib.Compose([
#             transform_lib.Resize(script_args.image_size),
#             transform_lib.CenterCrop(script_args.image_size),
#             transform_lib.ToTensor(),
#             transform_lib.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])
#         dataset = LSUN(root=script_args.data_dir, classes=["bedroom_train"], transform=transforms)
#         image_channels = 3
#     elif script_args.dataset == "mnist":
#         transforms = transform_lib.Compose([
#             transform_lib.Resize(script_args.image_size),
#             transform_lib.ToTensor(),
#             transform_lib.Normalize((0.5, ), (0.5, )),
#         ])
#         dataset = MNIST(root=script_args.data_dir, download=True, transform=transforms)
#         image_channels = 1

#     dataloader = DataLoader(
#         dataset, batch_size=script_args.batch_size, shuffle=True, num_workers=script_args.num_workers
#     )

#     parser = DCGAN.add_model_specific_args(parser)
#     parser = Trainer.add_argparse_args(parser)
#     args = parser.parse_args(args)

#     model = DCGAN(**vars(args), image_channels=image_channels)
#     callbacks = [
#         TensorboardGenerativeModelImageSampler(num_samples=5),
#         LatentDimInterpolator(interpolate_epoch_interval=5),
#     ]
#     trainer = Trainer.from_argparse_args(args, callbacks=callbacks)
#     trainer.fit(model, dataloader)

if __name__ == "__main__":
    cli_main()
