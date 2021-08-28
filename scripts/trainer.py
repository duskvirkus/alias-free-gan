from argparse import ArgumentParser
import sys
import os
import re
import subprocess

from torch.utils import data
from torchvision import transforms

import pytorch_lightning as pl
# from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

from utils.create_results_dir import create_results_dir
from utils.get_pretrained import get_pretrained_model_from_name, ModelNameNotFoundException
from utils.KimgSaverCallback import KimgSaverCallback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.stylegan2.dataset import MultiResolutionDataset

def modify_trainer_args(args):
    print(type(args))
    print(args)
    args['checkpoint_callback'] = False
    return args

def cli_main(args=None):

    print('Using Alias-Free GAN version: %s' % __version__)

    results_dir = create_results_dir()

    parser = ArgumentParser()

    script_parser = parser.add_argument_group("Trainer Script")
    script_parser.add_argument("--dataset_path", help='Path to dataset. Required!', type=str, required=True)
    script_parser.add_argument("--resume_from", help='Resume from checkpoint or transfer learn off pretrained model. Leave blank to train from scratch.', type=str, default=None)

    parser = AliasFreeGAN.add_model_specific_args(parser)
    parser = KimgSaverCallback.add_callback_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(args)

    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    os.makedirs(os.path.join(project_root, 'pretrained'), exist_ok=True)
    resume_path = None
    model_architecture = 'alias-free-rosinality-v1'

    custom_resume = args.resume_from.endswith('.pt')

    kimg_start_from_resume = None

    if custom_resume:
        print('Resuming from custom checkpoint...')

        if args.resume_from is not None:
            resume_path = args.resume_from
            a = re.search('[0-9]{9}', args.resume_from)
            if a:
                kimg_start_from_resume = int(a.group(0))
    else:
        try:
            pretrained = get_pretrained_model_from_name(args.resume_from)

            if pretrained.model_size != args.size:
                raise Exception(f'{pretrained.model_name} size of {pretrained.model_size} is not the same as size of {args.size} that was specified in arguments.')

            resume_path = pretrained.model_path
            model_architecture = pretrained.model_architecture

        except ModelNameNotFoundException as e:
            print(f'Warning! "{args.resume_from}" not found. Starting training from scratch.', flush=True)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    model_name = args.dataset_path.split('/')[-1] # set model name based on dataset name

    print(f'Dataset path: %s' % args.dataset_path)
    
    dataset = MultiResolutionDataset(args.dataset_path, transform=transform, resolution=args.size)
    
    print(f'Initialized {dataset.__class__.__name__} dataset with {dataset.__len__()} images')

    train_loader = data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    if args.resume_from_checkpoint:
        print('--resume_from_checkpoint is for pytorch lightning checkpoints. Please use --resume_from instead.')
        exit(1)

    kimg_callback = KimgSaverCallback(
        results_dir,
        model_name = model_name,
        kimg_start_from_resume = kimg_start_from_resume,
        **vars(args)
    )

    callbacks = [
        # TensorboardGenerativeModelImageSampler(num_samples=5),
        kimg_callback,
    ]

    trainer = pl.Trainer(
        logger = args.logger,
        checkpoint_callback=False,
        callbacks = callbacks,
        default_root_dir = args.default_root_dir,
        gradient_clip_val = args.gradient_clip_val,
        gradient_clip_algorithm = args.gradient_clip_algorithm,
        process_position = args.process_position,
        num_nodes = args.num_nodes,
        num_processes= args.num_processes,
        devices = args.devices,
        gpus = args.gpus,
        auto_select_gpus = args.auto_select_gpus,
        tpu_cores = args.tpu_cores,
        ipus = args.ipus,
        log_gpu_memory = args.log_gpu_memory,
        progress_bar_refresh_rate = args.progress_bar_refresh_rate,
        overfit_batches = args.overfit_batches,
        track_grad_norm = args.track_grad_norm,
        check_val_every_n_epoch = args.check_val_every_n_epoch,
        fast_dev_run = args.fast_dev_run,
        accumulate_grad_batches = args.accumulate_grad_batches,
        max_epochs = args.max_epochs,
        min_epochs = args.min_epochs,
        max_steps = args.max_steps,
        min_steps = args.min_steps,
        max_time = args.max_time,
        limit_train_batches = args.limit_train_batches,
        limit_val_batches = args.limit_val_batches,
        limit_test_batches = args.limit_test_batches,
        limit_predict_batches = args.limit_predict_batches,
        val_check_interval = args.val_check_interval,
        flush_logs_every_n_steps = args.flush_logs_every_n_steps,
        log_every_n_steps = args.log_every_n_steps,
        accelerator = args.accelerator,
        sync_batchnorm = args.sync_batchnorm,
        precision = args.precision,
        weights_summary= args.weights_summary,
        weights_save_path = args.weights_save_path,
        num_sanity_val_steps = args.num_sanity_val_steps,
        truncated_bptt_steps = args.truncated_bptt_steps,
        resume_from_checkpoint = args.resume_from_checkpoint,
        profiler = args.profiler,
        benchmark = args.benchmark,
        deterministic = args.deterministic,
        reload_dataloaders_every_n_epochs = args.reload_dataloaders_every_n_epochs,
        reload_dataloaders_every_epoch = args.reload_dataloaders_every_epoch,
        auto_lr_find = args.auto_lr_find,
        replace_sampler_ddp = args.replace_sampler_ddp,
        terminate_on_nan = args.terminate_on_nan,
        auto_scale_batch_size = args.auto_scale_batch_size,
        prepare_data_per_node = args.prepare_data_per_node,
        plugins = args.plugins,
        amp_backend = args.amp_backend,
        amp_level = args.amp_level,
        distributed_backend = args.distributed_backend,
        move_metrics_to_cpu = args.move_metrics_to_cpu,
        multiple_trainloader_mode = args.multiple_trainloader_mode,
        stochastic_weight_avg = args.stochastic_weight_avg,
    )

    model = AliasFreeGAN(model_architecture, resume_path, results_dir, kimg_callback, **vars(args))
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    cli_main()
