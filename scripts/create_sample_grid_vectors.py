from argparse import ArgumentParser

import torch

import numpy as np

def cli_main(args=None):

    parser = ArgumentParser()

    script_parser = parser.add_argument_group("Create Sample Grid Vectors Script")
    script_parser.add_argument('--rows', help='Number of rows in sample grid (default: %(default)s)', type=int, default=3)
    script_parser.add_argument('--cols', help='Number of columns in sample grid (default: %(default)s)', type=int, default=5)
    script_parser.add_argument('--seed', help='Random seed to use (default: %(default)s)', type=int, default=0)
    script_parser.add_argument('--style_dim', help='Style dimension size. (Not the same as model resolution, you\'ll proably know if you have to change this.) (default: %(default)s)', type=int, default=512)
    script_parser.add_argument('--include_zero_point_five_vec', help='Include vector with 0.5 for every dimension. Will be put in 0, 0 spot on the grid. (default: %(default)s)', type=bool, default=True)
    script_parser.add_argument('--save_location', help='Where the sample grid vectors will be saved.', type=str, required=True)
    args = parser.parse_args(args)

    sample_grid_vecs = []

    num_vectors_remaining = args.rows * args.cols

    if args.include_zero_point_five_vec:
        vec = []
        for i in range(args.style_dim):
            vec.append(0.5)
        sample_grid_vecs.append(np.reshape(vec, (1, args.style_dim)))
        num_vectors_remaining -= 1

    for i in range(num_vectors_remaining):
        sample_grid_vecs.append(np.random.RandomState(args.seed + i).randn(1, args.style_dim))

    torch.save({
        "sample_grid_rows": args.rows,
        "sample_grid_cols": args.cols,
        "sample_grid_vectors": sample_grid_vecs
    }, args.save_location)

if __name__ == "__main__":
    cli_main()
