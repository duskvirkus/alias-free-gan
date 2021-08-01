import numpy as np
from scipy.spatial import distance
from scipy.interpolate import interp1d

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
    

def circular(seed, frames, diameter):
    print('Generating %d frames for circular interpolation with seed %d.' % (frames, seed))

def simplex_noise(seed, frames):
    pass