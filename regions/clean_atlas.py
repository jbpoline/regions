from __future__ import print_function
import numpy as np
import scipy.ndimage.morphology as morph
import sys
import os
import tempfile

BACKGROUND = -1

def translate_struct(point, struct):
    """
    """
    return (np.asarray(np.where(struct)).T + np.asarray(point)-1).T            

def neigh_coords(point, struct, dim):
    """
    Get the coordinates of the points in the structure struct centered on point

    Parameters
    ----------
    point : array_like 
    struct : array_like 
        Structuring element
    dim : array_like
        Dimension (shape) of the domain 

    Returns
    -------
    neigh_coords : array of points of shape (points rank, number of points)

    Examples
    --------
    """

    translated = translate_struct(point, struct)
    # keep positive coordinates
    keep_pos = translated >= 0
    # keep coordinates within the domain of the image (dim)
    keep_small_than_dim = (translated.T - dim).T.max(axis=0) < 0
    # both of these 
    keep_both = np.logical_and(keep_pos, keep_small_than_dim).all(axis=0)
    keep = np.where(keep_both)[0]

    return translated[:,keep]



def neigh_values(point, struct, arr, exclude_centre=False):
    """
    Get the values of arr of the points in the structure struct centered on
    point

    Parameters
    ----------
    point : array_like 
    struct : array_like 
        Structuring element
    arr: array_like
        array from which the values are obtained 
    exclude_centre: bool
        if True will exclude the centre of the point at the centre of struct 

    Returns
    -------
    neigh_values : values at the position of the points in the struct centered
    on point 

    Examples
    --------
    """
    
    coords = neigh_coords(point, struct, arr.shape)
    if exclude_centre: 
        coords = rm_point_from_coords(point, coords)

    fancy_index = [coords[ii] for ii in range(coords.shape[0])]
    return arr[fancy_index]

def point_is_in_coords(point, coords):
    """
    check that point is in coords

    Parameters
    ----------
    point : array_like of shape (ndim, )
    coords: array_like of shape (ndim, nb_of_points)

    Returns
    -------
    bool

    Examples
    --------
    """
    try:
        assert np.asarray(point).shape[0] == np.asarray(coords).shape[0]
    except:
        raise ValueError('shape of point and coords incompatible')

    return np.any(~(np.asarray(coords).T - np.asarray(point)).any(axis=1))

def rm_point_from_coords(point, coords):
    """
    rm that point from coords

    Parameters
    ----------
    point : array_like of shape (ndim, )
    coords: array_like of shape (ndim, nb_of_points)

    Returns
    -------
    bool

    Examples
    --------
    """
    try:
        assert np.asarray(point).shape[0] == np.asarray(coords).shape[0]
    except:
        raise ValueError('shape of point and coords incompatible')

    idx2keep = (np.asarray(coords).T - np.asarray(point)).any(axis=1)
    return coords[:, idx2keep]


def get_border_coords(arr, struct, label=True):
    """
    check that point is in coords

    Parameters
    ----------
    arr: array_like 
    struct : array_like 
        Structuring element

    Returns
    -------
    coords: array of shape (3,n_coordinates)

    Examples
    --------
    """
   
    tmp = (arr==label)
    eroded = morph.binary_erosion(tmp, struct)

    return np.where(tmp - eroded)

def fill_hole_mult_label(arr, mask, struct, exclude=None):
    """
    fill points in mask (bool array indicating which regions have to be
    filled) with labels in arr

    Parameters
    ----------
    arr: array_like 
    mask: array_like 
    struct : array_like 
        Structuring element
    exclude: None, or number
        if number, will not consider this number when choosing label

    Returns
    -------
    filledarray : array of shape arr.shape

    Examples
    --------
    """
    filledarray = arr.copy()

    border = np.asarray(get_border_coords(mask, struct, label=1))
    if len(border[0]) == 0:
        return arr

    for cooidx in range(border.shape[1]):
        point = border[:,cooidx]
        neighvals = np.asarray(
                neigh_values(point, struct, arr, exclude_centre=True))
        
        # value most present ?
        uniquevals = np.unique(neighvals)

        if exclude is not None:
            uniquevals = [v for v in uniquevals if v != exclude]
            # if nothing around is not excluded (hole in background?)
            if len(uniquevals) == 0:
                filledarray[point] = exclude
                mask[point] = 0

        count = [(neighvals==v).sum() for v in uniquevals]
        label = uniquevals[np.argmax(count)]
        filledarray[point] = label
        mask[point] = 0

    # we have filled one border. Start again with the eroded mask!
    arr = filledarray
    return fill_hole_mult_label(arr, mask, struct, exclude=None)



        





















