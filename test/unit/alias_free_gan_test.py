import sys
import os
from argparse import ArgumentParser

import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from src.alias_free_gan import AliasFreeGAN

parser = ArgumentParser()

parser = AliasFreeGAN.add_model_specific_args(parser)

args = parser.parse_args(args)

print(args)

