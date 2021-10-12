import os
import sys
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

import numpy as np

import cv2

import PIL.Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.fake_dataloader import get_fake_dataloader

from utils.get_pretrained import get_pretrained_model_from_name

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Return a sharpened version of the image, using an unsharp mask.
    
    source: https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def cli_main(args=None):

    print('Using Alias-Free GAN version: %s' % __version__)

    parser = ArgumentParser()

    script_parser = parser.add_argument_group("Generate Script")
    script_parser.add_argument("--load_model", help='Load a model checkpoint to use for generating content.', type=str, required=True)
    script_parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True)
    script_parser.add_argument('--model_arch', help='The model architecture of the model to be loaded. (default: %(default)s)', type=str, default='alias-free-rosinality-v1')
    script_parser.add_argument('--seed', type=int, help='Seed value. (default: %(default)s)', default=0)
    script_parser.add_argument('--samples', type=int, help="Number of images with micro translations to generate. (default %(default)s", default=32)
    script_parser.add_argument('--trunc', type=float, help='Truncation psi (default: %(default)s)', default=0.75)
    script_parser.add_argument('--max_translate', type=float ,help="Max translation value. (default: %(default)s)", default=1.0)
    script_parser.add_argument('--batch', default=8, help='Number of images to generate each batch. default: %(default)s)')
    script_parser.add_argument('--upscale', help='Upscale factor. default: %(default)s)', default=4.0, type=float)
    parser = AliasFreeGAN.add_generate_specific_args(parser)

    args = parser.parse_args(args)

    trainer = pl.Trainer(gpus=1, max_epochs=0, log_every_n_steps=1)
    model = AliasFreeGAN(args.model_arch, args.load_model, args.outdir, None, **vars(args))
    trainer.fit(model, get_fake_dataloader(args.size))

    custom_checkpoint = args.load_model.endswith('.pt')

    if custom_checkpoint:
        print(f'Loading Custom Model from: {args.load_model}')
        model.load_checkpoint(args.load_model)
    else:
        print(f'Attempting to load pretrained model...')
        pretrained = get_pretrained_model_from_name(args.load_model)

        if pretrained.model_size != args.size:
            raise Exception(f'{pretrained.model_name} size of {pretrained.model_size} is not the same as size of {args.size} that was specified in arguments.')

        if args.model_arch != pretrained.model_architecture:
            raise Exception(f'Pretrained model_architecture of {pretrained.model_architecture} does not match --model_arch value of {args.model_arch}.')

        print(f'Loading pretrained model from: {pretrained.model_path}')
        model.load_checkpoint(pretrained.model_path)

        print(f'\n\n{pretrained.model_name} information:\n{pretrained.description}\n\n')

    os.makedirs(args.outdir, exist_ok=True)

    model.g_ema.eval()

    latent = torch.from_numpy(np.random.RandomState(args.seed).randn(1, model.generator.style_dim)).float().to(model.device)
    mean_latent = model.generator.mean_latent(4096)

    transform_p = model.generator.get_transform(
        latent, truncation=args.trunc, truncation_latent=mean_latent
    )

    M = np.eye(3, 3, dtype=np.float32)
    stacked_image = None
    first_sample = None

    with torch.no_grad():
        for i in range(args.samples):

            t_x = 0.0
            t_y = 0.0
            if i != 0:
                t_x = (np.random.random_sample() * 16) - 8
                t_y = (np.random.random_sample() * 16) - 8

            transform_p[:, 2] = t_y
            transform_p[:, 3] = t_x

            img = model.g_ema(latent, args.trunc, mean_latent, transform_p)

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img[0].cpu().numpy()

            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            if i == 0:

                cv2.imwrite(f'{args.outdir}/raw-seed{args.seed:04d}.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


            img = img.astype(np.float32) / 255

            img_shape = img.shape
            new_width = int(img_shape[1] * args.upscale)
            new_height = int(img_shape[0] * args.upscale)

            img = cv2.resize(img, (new_width, new_height), cv2.INTER_NEAREST)

            if i == 0:

                stacked_image = img

            else:

                # translate and add to stack
                img = cv2.warpAffine(
                    img,
                    np.float32([
	                    [1, 0, -t_x],
	                    [0, 1, -t_y]
                    ]),
                    (
                        img.shape[1],
                        img.shape[0]
                    )
                )

                stacked_image += img

            print(f'{i + 1}/{args.samples}')

        stacked_image /= float(args.samples)
        stacked_image = (stacked_image*255).astype(np.uint8)

        cv2.imwrite(f'{args.outdir}/super-res-seed{args.seed:04d}.png', stacked_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        sharpened = unsharp_mask(stacked_image, sigma=1.2, amount=1.5)
        cv2.imwrite(f'{args.outdir}/super-res-sharpened-seed{args.seed:04d}.png', sharpened, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/seed{args.seed:04d}-sample{i:04d}.png')

if __name__ == "__main__":
    cli_main()

# def stackImagesECC(file_list):
#     M = np.eye(3, 3, dtype=np.float32)

#     first_image = None
#     stacked_image = None

#     for file in file_list:
#         image = cv2.imread(file,1).astype(np.float32) / 255
#         print(file)
#         if first_image is None:
#             # convert to gray scale floating point image
#             first_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#             stacked_image = image
#         else:
#             # Estimate perspective transform
#             s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)
#             w, h, _ = image.shape
#             # Align image to first image
#             image = cv2.warpPerspective(image, M, (h, w))
#             stacked_image += image

#     stacked_image /= len(file_list)
#     stacked_image = (stacked_image*255).astype(np.uint8)
#     return stacked_image

# import os
# import sys
# from argparse import ArgumentParser

# import pytorch_lightning as pl

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# from src import __version__
# from src.alias_free_gan import AliasFreeGAN
# from src.fake_dataloader import get_fake_dataloader

# from utils.get_pretrained import get_pretrained_model_from_name

# def cli_main(args=None):

#     print('Using Alias-Free GAN version: %s' % __version__)

#     parser = ArgumentParser()

#     script_parser = parser.add_argument_group("Generate Script")
#     script_parser.add_argument("--load_model", help='Load a model checkpoint to use for generating content.', type=str, required=True)
#     script_parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True)
#     script_parser.add_argument('--model_arch', help='The model architecture of the model to be loaded. (default: %(default)s)', type=str, default='alias-free-rosinality-v1')
#     script_parser.add_argument('--seed_start', type=int, help='Start range for seed values. (default: %(default)s)', default=0)
#     script_parser.add_argument('--seed_stop', type=int, help='Stop range for seed values. Is inclusive. (default: %(default)s)', default=99)
#     script_parser.add_argument('--trunc', type=float, help='Truncation psi (default: %(default)s)', default=0.75)
#     script_parser.add_argument('--batch', default=8, help='Number of images to generate each batch. default: %(default)s)')
#     parser = AliasFreeGAN.add_generate_specific_args(parser)

#     args = parser.parse_args(args)

#     trainer = pl.Trainer(gpus=1, max_epochs=0, log_every_n_steps=1)
#     model = AliasFreeGAN(args.model_arch, args.load_model, args.outdir, None, **vars(args))
#     trainer.fit(model, get_fake_dataloader(args.size))

#     custom_checkpoint = args.load_model.endswith('.pt')

#     if custom_checkpoint:
#         print(f'Loading Custom Model from: {args.load_model}')
#         model.load_checkpoint(args.load_model)
#     else:
#         print(f'Attempting to load pretrained model...')
#         pretrained = get_pretrained_model_from_name(args.load_model)

#         if pretrained.model_size != args.size:
#             raise Exception(f'{pretrained.model_name} size of {pretrained.model_size} is not the same as size of {args.size} that was specified in arguments.')

#         if args.model_arch != pretrained.model_architecture:
#             raise Exception(f'Pretrained model_architecture of {pretrained.model_architecture} does not match --model_arch value of {args.model_arch}.')

#         print(f'Loading pretrained model from: {pretrained.model_path}')
#         model.load_checkpoint(pretrained.model_path)

#         print(f'\n\n{pretrained.model_name} information:\n{pretrained.description}\n\n')

#     seeds = []
#     for i in range(args.seed_start, args.seed_stop, 1):
#         seeds.append(i)

#     model.generate_images(seeds, args.outdir, args.trunc)

# if __name__ == "__main__":
#     cli_main()


# # Based on Rosinality generate.py
# # https://github.com/rosinality/alias-free-gan-pytorch/commit/9dfd1255823dab98608edb7d25e7f81cab05b6ce

# import os
# import sys
# from argparse import ArgumentParser
# import inspect
# import random

# import torch
# from torchvision import utils

# import pytorch_lightning as pl

# import numpy as np

# import cv2

# from tqdm import tqdm

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# from src import __version__
# from src.alias_free_gan import AliasFreeGAN
# from src.fake_dataloader import get_fake_dataloader

# import utils.easings as easings
# import interpolation.methods as methods
# from utils.get_pretrained import get_pretrained_model_from_name

# def cli_main(args=None):

#     print('Using Alias-Free GAN version: %s' % __version__)

#     parser = ArgumentParser()
#     parser = AliasFreeGAN.add_generate_specific_args(parser)

#     script_parser = parser.add_argument_group("Rosinality Generate Script")

#     script_parser.add_argument('--model_arch', help='The model architecture of the model to be loaded. (default: %(default)s)', type=str, default='alias-free-rosinality-v1')
#     script_parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True)
#     script_parser.add_argument('--batch', default=8, help='Not implymented yet! Number of images to generate each batch. default: %(default)s)') #TODO currently does nothing

#     script_parser.add_argument(
#         "--n_img", type=int, default=16, help="number of images to be generated (default: %(default)s)"
#     )
#     script_parser.add_argument(
#         "--n_row", type=int, default=4, help="number of samples per row (default: %(default)s)"
#     )
#     script_parser.add_argument(
#         "--truncation", type=float, default=0.5, help="truncation ratio (default: %(default)s)"
#     )
#     script_parser.add_argument(
#         "--truncation_mean",
#         type=int,
#         default=4096,
#         help="number of vectors to calculate mean for the truncation (default: %(default)s)",
#     )
#     script_parser.add_argument("--n_frame", help="(default: %(default)s)", type=int, default=120)
#     script_parser.add_argument("--radius", help="(default: %(default)s)", type=float, default=30)
#     script_parser.add_argument("--video", help="Export a video from frames. (default: %(default)s)", type=bool, default=True)
#     script_parser.add_argument("--frames", help="Save frames in frames subfolder. (default: %(default)s)", type=bool, default=False)
#     script_parser.add_argument(
#         "ckpt", metavar="CKPT", type=str, help="path to the model checkpoint"
#     )

#     args = parser.parse_args(args)

#     trainer = pl.Trainer(gpus=1, max_epochs=0, log_every_n_steps=1)
#     model = AliasFreeGAN(args.model_arch, args.ckpt, args.outdir, None, **vars(args))
#     trainer.fit(model, get_fake_dataloader(args.size))

#     custom_checkpoint = args.ckpt.endswith('.pt')

#     if custom_checkpoint:
#         print(f'Loading Custom Model from: {args.ckpt}')
#         model.load_checkpoint(args.ckpt)
#     else:
#         print(f'Attempting to load pretrained model...')
#         pretrained = get_pretrained_model_from_name(args.ckpt)

#         if pretrained.model_size != args.size:
#             raise Exception(f'{pretrained.model_name} size of {pretrained.model_size} is not the same as size of {args.size} that was specified in arguments.')

#         if args.model_arch != pretrained.model_architecture:
#             raise Exception(f'Pretrained model_architecture of {pretrained.model_architecture} does not match --model_arch value of {args.model_arch}.')

#         print(f'Loading pretrained model from: {pretrained.model_path}')
#         model.load_checkpoint(pretrained.model_path)

#         print(f'\n\n{pretrained.model_name} information:\n{pretrained.description}\n\n')

#     model.generator.eval()

#     mean_latent = model.generator.mean_latent(args.truncation_mean)
#     x = torch.randn(args.n_img, model.generator.style_dim, device=model.device)

#     theta = np.radians(np.linspace(0, 360, args.n_frame))
#     x_2 = np.cos(theta) * args.radius
#     y_2 = np.sin(theta) * args.radius

#     trans_x = x_2.tolist()
#     trans_y = y_2.tolist()

#     images = []

#     transform_p = model.generator.get_transform(
#         x, truncation=args.truncation, truncation_latent=mean_latent
#     )

#     with torch.no_grad():
#       for i, (t_x, t_y) in enumerate(tqdm(zip(trans_x, trans_y), total=args.n_frame)):
#           transform_p[:, 2] = t_y
#           transform_p[:, 3] = t_x

#           img = model.generator(
#               x,
#               truncation=args.truncation,
#               truncation_latent=mean_latent,
#               transform=transform_p,
#           )
#           images.append(
#               utils.make_grid(
#                   img.cpu(), normalize=True, nrow=args.n_row, value_range=(-1, 1)
#               )
#               .mul(255)
#               .permute(1, 2, 0)
#               .numpy()
#               .astype(np.uint8)
#           )
#     videodims = (images[0].shape[1], images[0].shape[0])

#     base_name = 'ros-gen-' + args.ckpt.split('/')[-1]

#     os.makedirs(args.outdir, exist_ok=True)

#     if args.frames:

#         os.makedirs(os.path.join(args.outdir, 'frames'), exist_ok=True)

#         for i in range(len(images)):
#             im = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)

#             new_file = base_name + '-' + str(i).zfill(9) + ".png"
#             cv2.imwrite(os.path.join(args.outdir, 'frames', new_file), im, [cv2.IMWRITE_PNG_COMPRESSION, 0])

#     if args.video:

#         vid_name = base_name + '.mp4'

#         fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#         vid = cv2.VideoWriter(os.path.join(args.outdir, vid_name), fourcc, 24, videodims)

#         for i in tqdm(images):
#             vid.write(cv2.cvtColor(i, cv2.COLOR_RGB2BGR))

#         vid.release()

# if __name__ == "__main__":
#     cli_main()