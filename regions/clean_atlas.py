from __future__ import print_function
import numpy as np
import sys
import os
import tempfile

def translate_struct(point, struct):
    """
    """
    return (np.asarray(np.where(struct)).T + np.asarray(point)-1).T            

def neigh_coords(point, struct, dim):
    translated = translate_struct(point, struct)
    # keep positive coordinates
    keep_pos = translated >= 0
    # keep coordinates within the domain of the image (dim)
    keep_small_than_dim = (translated.T - dim).T.max(axis=0) < 0
    # both of these 
    keep_both = np.logical_and(keep_pos, keep_small_than_dim).all(axis=0)
    keep = np.where(keep_both)[0]

    return translated[:,keep]



def neigh_values(point, struct, arr):
    """
    """
    coords = neigh_coords(point, struct, arr.shape)
    fancy_index = [coords[ii] for ii in range(coords.shape[0])]
    return arr[fancy_index]




