from __future__ import print_function
import numpy as np
import scipy.ndimage.morphology as morph
import sys
import os
import tempfile
from datetime import datetime 
import logging as logg

logg.basicConfig(level=logg.DEBUG)
LOG = logg.getLogger("cleanAtlas")

dtime = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
print("reloading clean atlas at %s \n" % dtime)
sys.stdout.flush()


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

def idxof(coo):
    """
    transform simple coo to allow indexing
    """
    coo = np.asarray(coo).flatten()
    l = [slice(c,c+1,1) for c in coo]
    return tuple(l)

def valof(arr, coo):
    """
    get value of arr at coo
    """
    return arr[tuple(coo)]




def log(msg, *args):
    # print(" LOG.setLevel : {}", LOG.setLevel)
    # print(" Effective : {}", LOG.getEffectiveLevel())
    to_print = print(msg.format(*args))
    LOG.log(LOG.setLevel, to_print, ()) # msg, *args)

def fill_hole_mult_label(arr, mask, struct, exclude=None, loglevel=0):
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
    maskcopy = mask.copy()
    LOG.setLevel = loglevel

    log("fill_holes loglevel : {} Effective {} ", 
            LOG.setLevel, LOG.getEffectiveLevel())
    log("\n\n ======================== Entering ================== ")
    log("mask.sum {} ", mask.sum())
    log(" mask {}", mask)
    log(" arr {}",arr)

    if mask.sum() == 0: #nothing in mask
        return arr

    # coords_in_mask = np.where(mask)
    border = np.asarray(get_border_coords(mask, struct, label=True))

    log("border.shape = {} ", border.shape) 

    nb_pts = border.shape[1]
    #ndim = border.shape[0]

    for cooidx in range(nb_pts):
        point = border[:,cooidx]
        log("\n ======================== point is {}", point) 
        no_progress = True
        neighcoords = neigh_coords(point, struct, arr.shape)
        # keep these coords that are not in the mask (the hole)
        neighcoords = np.asarray([pt for pt in neighcoords.T 
                                            if mask[tuple(pt)] == False]).T
        log("neighcoords :{} \n",neighcoords)
        log("arr val {}", [arr[tuple(pt)] for pt in neighcoords.T])
        log("mask val {}", [mask[tuple(pt)] for pt in neighcoords.T])

        neighvals = np.asarray([arr[tuple(pt)] for pt in neighcoords.T])
        log('neighvals {}', (point, neighvals))
        # value most present ?
        uniquevals = np.unique(neighvals)
        # log('point uniquevals ', (point, uniquevals), file=sys.stdout)

        if exclude is not None:
            uniquevals = [v for v in uniquevals if (v != exclude)]
            # if nothing around is not excluded (hole in background?)
            if len(uniquevals) == 0:
                log("*** no non excluded point in environment of : {} ", point)
                #filledarray[point] = exclude
                #mask[point] = 0
                continue

        log('uniquevals without excluded {} ', uniquevals) 

        count = [(neighvals==v).sum() for v in uniquevals]
        log("\t count {} ", count)
        label = uniquevals[np.argmax(count)]
        filledarray[tuple(point)] = label
        log("\t ----- putting label {} ------------", label)
        maskcopy[tuple(point)] = False

    # we have filled one border. Start again with the eroded mask!
    arr = filledarray
    mask = maskcopy
    return fill_hole_mult_label(arr, mask, struct, 
                                    exclude=exclude, loglevel=loglevel)



        





