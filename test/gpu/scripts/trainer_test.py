import subprocess
import os
import gc

import torch

import pytest

def clean_up():
    gc.collect()
    torch.cuda.empty_cache() 

def test_trainer_from_scratch():
    trainer_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'scripts/trainer.py')
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'alias-free-gan-ci-files/flowers-test-dataset-32-256')
    p = subprocess.run(["python", trainer_script_path, "--size", "256", "--gpus", "1", "--dataset_path", dataset_path, "--batch", "8", "--max_epochs", "1"])
    assert p.returncode == 0
    clean_up()

def test_trainer_from_ffhq():
    trainer_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'scripts/trainer.py')
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'alias-free-gan-ci-files/flowers-test-dataset-32-256')
    p = subprocess.run(["python", trainer_script_path, "--size", "256", "--gpus", "1", "--dataset_path", dataset_path, "--batch", "8", "--max_epochs", "1", "--resume_from", "rosinality-ffhq-800k"])
    assert p.returncode == 0
    clean_up()

def test_trainer_from_checkpoint():
    trainer_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'scripts/trainer.py')
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'alias-free-gan-ci-files/flowers-test-dataset-32-256')
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'alias-free-gan-ci-files/000000001-kimg-ci-checkpoint.pt')
    p = subprocess.run(["python", trainer_script_path, "--size", "256", "--gpus", "1", "--dataset_path", dataset_path, "--batch", "8", "--max_epochs", "1", "--resume_from", checkpoint_path])
    assert p.returncode == 0
    clean_up()

def test_trainer_with_augmentation():
    trainer_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'scripts/trainer.py')
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'alias-free-gan-ci-files/flowers-test-dataset-32-256')
    p = subprocess.run(["python", trainer_script_path, "--size", "256", "--gpus", "1", "--dataset_path", dataset_path, "--batch", "8", "--max_epochs", "1", '--augment', 'True'])
    assert p.returncode == 0
    clean_up()
