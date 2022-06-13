import math
import numpy as np

def nplanes_tab(n_scales, n_layers, nplanes_init, nplanes_mulv,
                nplanes_addh, prune, plane_reduction):
    """Tracks number of channels at each layer and scale"""
    # Identifies which layers to reduce scale
    reduce_layers = _reduce_layers(n_scales, n_layers, prune,
                                        plane_reduction)
    print(reduce_layers)
    # Identifies which scales are not needed (in final layer)
    nprune_per_layer = _nprune_per_layer(n_scales, n_layers, prune)
    print(nprune_per_layer)
    # Creates the large array containing every channel at each layer (column) and scale (row)
    hbase, nprune = [nplanes_init], [0]
    for i in range(1, n_layers):
        hbase += [hbase[-1] + nplanes_addh]
        nprune += [nprune_per_layer[i]]
        if i in reduce_layers:
            # Not sure this behavior is intended, especially if the scale isn't removed!
            # (Currently with prune = 'min' scales do not get removed at the intervals,
            # just have the channels reduced by half)
            hbase += [math.floor(hbase[-1] * plane_reduction)]
            nprune += [nprune_per_layer[i]]
    # Multiplies each scales (row) by the number of channels after input
    planes_tab = np.outer(nplanes_mulv, hbase)
    print(planes_tab)
    # Prunes the final scales which are unneeded
    for i in range(len(hbase)):
        planes_tab[:nprune[i], i] = 0
    return planes_tab

def _reduce_layers(n_scales, n_layers, prune, plane_reduction): # Works
    if not plane_reduction:
        return []
    # Only reduces scales twice at 1/3 and 2/3 interval
    elif prune == 'min':
        return [math.floor((n_layers-1)*1/3),
                math.floor((n_layers-1)*2/3)]
    # Reduces scales until only one scale is left
    elif prune == 'max':
        interval = math.ceil((n_layers-1) / n_scales)
        return list(range(interval+1, n_layers, interval))

def _nprune_per_layer(n_scales, n_layers, prune): # Works
    # Only prunes the unused scales in last block
    if prune == 'min':
        nprune = min(n_scales, n_layers) - np.arange(n_layers, 0, -1)
        return list(np.maximum(0, nprune))
    # Actually removes the scales at each interval
    elif prune == 'max':
        interval = math.ceil((n_layers-1) / n_scales)
        return [0] + [math.floor(i/interval) for i in range(n_layers-1)]
    else:
        return [0] * n_layers

in_shape = 32
out_dim = 10
n_scales = 3
n_exits = 3
nlayers_to_exit = 2
nlayers_between_exits = 2
nplanes_mulv = [2, 4, 8]
nplanes_addh = 1
nplanes_init = 1
prune = "max"
plane_reduction = 0.5
n_layers = 7

planes_tab = nplanes_tab(n_scales, n_layers, nplanes_init, nplanes_mulv,
                nplanes_addh, prune, plane_reduction)

print(planes_tab)
