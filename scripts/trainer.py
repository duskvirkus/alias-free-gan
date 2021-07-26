from argparse import ArgumentParser
import sys
import os

from torch.utils import data
from torchvision import transforms

import pytorch_lightning as pl
# from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

import gdown

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.alias_free_gan import AliasFreeGAN
from src.stylegan2.dataset import MultiResolutionDataset
from src.utils import sha1_hash
from src.pretrained_models import pretrained_models

def cli_main(args=None):

    parser = ArgumentParser()

    script_parser = parser.add_argument_group("Trainer Script")
    script_parser.add_argument("--dataset_path", help='Path to dataset. Required!', type=str, required=True)
    script_parser.add_argument("--resume_from", help='Resume from checkpoint or transfer learn off pretrained model. Leave blank to train from scratch.', type=str, default=None)

    parser = AliasFreeGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(args)

    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    resume_path = None
    for pretrained in pretrained_models:
        if args.resume_from == pretrained['model_name']:

            if args.size == pretrained['model_size']:
                save_path = os.path.join(project_root, 'pretrained', pretrained['model_name'] + '.pt')
                if not os.path.isfile(save_path):
                    print('Downloading %s from %s' % (pretrained['model_name'], pretrained['download_url']))
                    gdown.download(pretrained['download_url'], save_path, quiet=False)

                # verify hash
                sha1_hash_val = sha1_hash(save_path)
                if (sha1_hash_val != pretrained['sha1']):
                    print('Unexpected sha1 hash for %s! Expected %s but got %s. If you see this error try deleting %s and rerunning to download the pretrained model it indicates a corrupted file.' % (save_path, pretrained['sha1'], sha1_hash_val, save_path))
                
                resume_path = save_path
            else:
                print('Invalid model size for %s! Model works with size=%d but your trying to train a size=%d model.' % (pretrained['model_name'], pretrained['model_size'], args.size))
                exit(1)

    if args.resume_from is not None:
        resume_path = save_path

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

    model = AliasFreeGAN(resume_path, **vars(args))
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    cli_main()
