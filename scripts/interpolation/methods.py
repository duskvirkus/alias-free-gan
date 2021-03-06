import math

import numpy as np

import torch

from scipy.spatial import distance
from scipy.interpolate import interp1d

from opensimplex import OpenSimplex

def interpolate(style_dim, seeds, frames, easing_func):
    print('Generating %d frames for interpolation with seeds %s with %s easing.' % (frames, seeds, easing_func.__name__))

    fenceposts = []
    for seed in seeds:
        fenceposts.append(np.random.RandomState(seed).randn(1, style_dim))

    distances = []
    for i in range(len(fenceposts) - 1):
        distances.append(distance.euclidean(fenceposts[i], fenceposts[i + 1]))

    distance_sum = sum(distances)

    z_vectors = [fenceposts[0]]

    m = interp1d([0,frames], [0,distance_sum])
    for i in range(frames - 2):
        frame = i + 1
        location = m(frame)
        span_index = 0
        while location > distances[span_index]:
            location -= distances[span_index]
            span_index += 1
        m2 = interp1d([0,distances[span_index]], [0,1])
        amt = m2(location)
        eased_amt = easing_func(amt)
        vec = []
        for j in range(style_dim):
            m3 = interp1d([0,1], [fenceposts[span_index][0][j],fenceposts[span_index + 1][0][j]])
            vec.append(m3(eased_amt))

        vector = np.reshape(vec, (1, style_dim))
        z_vectors.append(vector)

    z_vectors.append(fenceposts[-1])
    return z_vectors
    

def circular(style_dim, seed, frames, diameter_list):
    print('Generating %d frames for circular interpolation with seed %d.' % (frames, seed))

    # TODO this could likely be improved

    range_value = 11

    for diameter in diameter_list:
        if diameter > (range_value - 1):
            print('Diameter %f is bigger than maximum of %f' % (diameter, range_value - 1))

    map_frame = interp1d([0, frames - 1], [0, math.pi * 2])
    circular_map = interp1d([-(range_value * range_value), range_value * range_value], [0, 1])

    position_offsets = np.random.RandomState(seed).rand(style_dim, 2)

    z_vectors = []
    for i in range(frames):
        angle = map_frame(i)

        vec = []

        for j in range(style_dim):
            x = position_offsets[j][0] + (diameter_list[j] / 2) * math.cos(angle)
            y = position_offsets[j][1] + (diameter_list[j] / 2) * math.sin(angle)
            vec.append(x * y)
        
        z_vectors.append(np.reshape(vec, (1, style_dim)))

    return z_vectors

def simplex_noise(style_dim, seed, frames, diameter_list):
    print('Generating %d frames for simplex noise interpolation with seed %d.' % (frames, seed))

    simplex = OpenSimplex(seed)
    map_frame = interp1d([0, frames - 1], [0, math.pi * 2])

    position_offsets = np.random.RandomState(seed).randn(style_dim, 2)

    z_vectors = []
    for i in range(frames):
        angle = map_frame(i)

        vec = []

        for j in range(style_dim):
            x_off = position_offsets[j][0] + (diameter_list[j] / 2) * math.cos(angle)
            y_off = position_offsets[j][1] + (diameter_list[j] / 2) * math.sin(angle)
            vec.append(simplex.noise2d(x_off, y_off))
        
        z_vectors.append(np.reshape(vec, (1, style_dim)))

    return z_vectors

def load_z_vectors(z_vectors_path):
    z_vectors_file = torch.load(z_vectors_path)
    return z_vectors_file['z_vectors']
