import sys
import os
from argparse import ArgumentParser

import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from src.alias_free_gan import AliasFreeGAN

class TestAliasFreeGAN():
    
    def test_add_model_specific_args_defaults(self):
        parser = ArgumentParser()
        parser = AliasFreeGAN.add_model_specific_args(parser)
        args = parser.parse_args(['--size', '256'])

        assert args.ada_every == 256
        assert args.ada_length == 500000
        assert args.ada_target == 0.6
        assert args.argument_p == 0.0
        assert args.augment == False
        assert args.batch == 16
        assert args.d_reg_every == 16
        assert args.lr_d == 0.002
        assert args.lr_g == 0.002
        assert args.n_samples == 9
        assert args.r1 == 10.0
        assert args.size == 256
