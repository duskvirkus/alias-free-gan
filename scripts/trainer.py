from argparse import ArgumentParser
import sys
import os
import json
import re
import subprocess

from torch.utils import data
from torchvision import transforms

import pytorch_lightning as pl
# from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.stylegan2.dataset import MultiResolutionDataset
from src.utils import sha1_hash
from src.pretrained_models import pretrained_models

from utils.KimgSaverCallback import KimgSaverCallback

def modify_trainer_args(args):
    print(type(args))
    print(args)
    args['checkpoint_callback'] = False
    return args

def load_pretrained_models():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pretrained_models.json')) as json_file:
        data = json.load(json_file)

        return data['pretrained_models']

def create_results_dir():
    results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')
    os.makedirs(results_root, exist_ok=True)

    max_num = -1

    for root, subdirs, files in os.walk(results_root):
        for subdir in subdirs:
            numbers = re.findall('[0-9]+', subdir)
            if numbers:
                if (int(numbers[0]) > max_num):
                    max_num = int(numbers[0])
    
    max_num += 1

    results_dir = os.path.join(results_root, 'training-' + str(max_num).zfill(6))
    os.makedirs(results_dir)
    return results_dir

def cli_main(args=None):

    print('Using Alias-Free GAN version: %s' % __version__)

    results_dir = create_results_dir()

    pretrained_models = load_pretrained_models()

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
    for pretrained in pretrained_models:
        if args.resume_from == pretrained['model_name']:

            if args.size == pretrained['model_size']:
                save_path = os.path.join(project_root, 'pretrained', pretrained['model_name'] + '.pt')
                if not os.path.isfile(save_path):
                    print('Downloading %s from %s' % (pretrained['model_name'], pretrained['wget_url']))
                    sys_call = subprocess.run(["wget", "--progress=bar:force", "-O", save_path, pretrained['wget_url']], capture_output=True)
                    print(sys_call)
                    if sys_call.returncode != 0:
                        exit(7)

                # verify hash
                sha1_hash_val = sha1_hash(save_path)
                if (sha1_hash_val != pretrained['sha1']):
                    print('Unexpected sha1 hash for %s! Expected %s but got %s. If you see this error try deleting %s and rerunning to download the pretrained model it indicates a corrupted file.' % (save_path, pretrained['sha1'], sha1_hash_val, save_path))
                    exit(6)
                
                resume_path = save_path
                model_architecture = pretrained['model_architecture']

                print('\n\nLicence and compensation information for %s pretrained model: %s\n\n' % (pretrained['model_name'], pretrained['licence_and_compensation_information']))
            else:
                print('Invalid model size for %s! Model works with size=%d but your trying to train a size=%d model.' % (pretrained['model_name'], pretrained['model_size'], args.size))
                exit(1)

    kimg_start_from_resume = None
    if resume_path is None and args.resume_from is not None:
        resume_path = args.resume_from
        a = re.search('[0-9]{9}', args.resume_from)
        if a:
            kimg_start_from_resume = int(a.group(0))

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
