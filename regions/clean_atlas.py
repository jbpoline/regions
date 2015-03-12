from __future__ import print_function
import numpy as np
import scipy.ndimage.morphology as morph
import sys
import os
import tempfile
from datetime import datetime 
import logging as logg

logg.basicConfig()
LOG = logg.getLogger("clean_atlas")

dtime = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
print("reloading clean atlas at %s \n" % dtime)
#sys.stdout.flush()


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

def critical(msg, *args): # 50
    log(logg.CRITICAL, msg, *args)

def error(msg, *args): # 40
    log(logg.ERROR, msg, *args)

def warn(msg, *args): # 30
    log(logg.WARN, msg, *args)

def info(msg, *args): # 20
    log(logg.INFO, msg, *args)

def debug(msg, *args): # 10
    log(logg.DEBUG, msg, *args)

def log(localLogLevel, msg, *args):
    toprnt = msg.format(*args)
    LOG.log(localLogLevel, toprnt)

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
    filled_array: array of shape arr.shape

    Examples
    --------
    """

    # set global level of logging 
    LOG.setLevel(loglevel)
    

    debug("fill_holes loglevel : {} Effective {} ", 
            LOG.setLevel, LOG.getEffectiveLevel())
    debug("\n\n ======================== Entering ================== ")
    debug("mask.sum {} ", mask.sum())
    debug(" mask {}", mask)
    debug(" arr {}",arr)

    hole_size = mask.sum()
    if hole_size == 0: #nothing in mask
        return arr

    progress = True
    filled_labels = []

    hole_size = mask.sum()
    while (hole_size > 0 and progress):
        arraycopy = arr.copy()
        maskcopy = mask.copy()
        progress = False # as usual
        border = np.asarray(get_border_coords(mask, struct, label=True))
        debug("Working on border with border.shape = {} ", border.shape) 
        nb_pts = border.shape[1]
        if nb_pts:
            hole_val = arr[tuple(border[:,0])]
        else:
            hole_val = None
        critical(" nb of points in border {} of hole size {} array val {}",
                nb_pts, hole_size, hole_val)
        #ndim = border.shape[0]
    
        #--------------- for the coordinates in the border ------------------#
        for cooidx in range(nb_pts):
            point = border[:,cooidx]
            debug("\n ======================== point is {}", point) 
            neighcoords = neigh_coords(point, struct, arr.shape)
    
            # keep these coords that are not in the mask, eg mask[pt] == False 
            neighcoords = np.asarray([pt for pt in neighcoords.T 
                if mask[tuple(pt)] == False]).T
    
            debug("neighcoords :{}",neighcoords)
            debug("arr val {}", [arr[tuple(pt)] for pt in neighcoords.T])
            debug("mask val {}", [mask[tuple(pt)] for pt in neighcoords.T])
    
            # get the values of these neighbors coords
            neighvals = np.asarray([arr[tuple(pt)] for pt in neighcoords.T])
            debug('neighvals {}', (point, neighvals))
    
            # what values of the neighbourhood are of interest ?
            uniquevals = np.unique(neighvals)
            debug('point {} uniquevals {} ', point, uniquevals)
    
            if exclude is not None:
                uniquevals = [v for v in uniquevals if (v != exclude)]
                # if all neighbour points are excluded (hole in background?)
                if len(uniquevals) == 0:
                    warn("*** all point around {} are excluded  ", point)
                    #arraycopy[point] = exclude
                    #mask[point] = 0
                    continue
    
            debug('uniquevals without excluded {} ', uniquevals) 
    
            # values most present around this point? 
            count = [(neighvals==v).sum() for v in uniquevals]
            debug("\t count {} ", count)
            label = uniquevals[np.argmax(count)]
    
            # a point is updated by a label
            debug("\t ----- putting label {} ------------", label)
            arraycopy[tuple(point)] = label
            maskcopy[tuple(point)] = False
            filled_labels.append(label)
            progress = True
        #--------------- for the coordinates in the border ------------------#

        # any progress after this border ?
        if progress:
            arr = arraycopy
            mask = maskcopy
            hole_size = mask.sum()
            critical(" fill with labels : {}", filled_labels)
        else:
            warn("*** ---- no progress made at this border -----  ")

    #------------ while ------------- #
    
    critical("--------  exit ----------------------- ") 
    return arr

        





