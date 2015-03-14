from __future__ import print_function
import numpy as np
import scipy.ndimage.morphology as morph
# import sys # import os import tempfile from datetime 
import datetime as dt 
import logging as logg

logg.basicConfig(level=logg.CRITICAL)
#hdlr = logg.StreamHandler(stream=sys.stdout)
LOG = logg.getLogger("clean_atlas")
#LOG.addHandler(hdlr)
#LOG.setLevel(logg.WARN)

dtime = dt.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
print("reloading clean atlas at %s \n" % dtime)


def translate_struct(point, struct):
    """
    np.asarray(np.where(struct)).T : list of points in n_pts X 3 shape
    + point for translation
    - 1 : the struct is centered on [1,1,1]
    """
    if len(struct.shape) != len(point):
        msg = "struc {} and point {} incompatible".format(struct, point) 
        raise TypeError, msg

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
    keep_smaller_than_dim = (translated.T - dim).T.max(axis=0) < 0
    # both of these 
    keep_both = np.logical_and(keep_pos, keep_smaller_than_dim).all(axis=0)
    # np.where returns a tuple ('array of indices', ) : get the array with [0]
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
    return a list of the points that form the border of
    points with label

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


# shortcut for logging functions
# def critical(msg, logger=LOG, *args): # 50
#     logger.critical(msg, *arg)
# def error(msg, logger=LOG, *args): # 40
#     logger.error(msg, *arg)
# def warn(msg, logger=LOG, *args): # 30
#     logger.warn(msg, *arg)
# def info(msg, logger=LOG, *args): # 20
#     logger.info(msg, *arg)
# def debug(msg, *args, **{'logger':LOG}): # 10
#     logger.debug(msg, *arg)

def fill_hole_mult_label(arr, mask, struct, exclude=None, logger=LOG, loglevel=logg.CRITICAL):
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
    # set global level of logging : lower value prints more
    logger.setLevel(loglevel)

    logger.debug("fill_holes loglevel : {} Effective {} ", logger.setLevel(loglevel), 
                                                    logger.getEffectiveLevel())
    logger.debug("\n\n ======================== Entering ================== ")
    logger.debug("mask.sum {} ", mask.sum())
    logger.debug(" mask {}", mask)
    logger.debug(" arr {}", arr)
    # print("logger.getEffectiveLevel()", logger.getEffectiveLevel())
    # sys.stdout.flush()

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
        
        #--------------- get the border of the mask ------------------#
        border = np.asarray(get_border_coords(mask, struct, label=True))
        logger.debug("Working on border with border.shape = {} ", border.shape) 
        nb_pts = border.shape[1]
        if nb_pts:
            hole_val = arr[tuple(border[:,0])]
        else:
            hole_val = None

        logger.critical(" nb of points in border {} of hole size {} array val {}",
                nb_pts, hole_size, hole_val)
    
        #--------------- for the coordinates in the border ------------------#
        for cooidx in range(nb_pts):
            point = border[:,cooidx]
            logger.debug("\n ======================== point is {}", point) 
            neighcoords = neigh_coords(point, struct, arr.shape)
    
            # keep these coords that are not in the mask, eg mask[pt] == False 
            neighcoords = np.asarray([pt for pt in neighcoords.T 
                                                if mask[tuple(pt)] == False]).T
    
            logger.debug("neighcoords :{}",neighcoords)
            logger.debug("arr val {}", [arr[tuple(pt)] for pt in neighcoords.T])
            logger.debug("mask val {}", [mask[tuple(pt)] for pt in neighcoords.T])
    
            # get the values of these neighbors coords
            neighvals = np.asarray([arr[tuple(pt)] for pt in neighcoords.T])
            logger.debug('neighvals {}', (point, neighvals))
    
            # what values of the neighbourhood are of interest ?
            uniquevals = np.unique(neighvals)
            logger.debug('point {} uniquevals {} ', point, uniquevals)
    
            if exclude is not None:
                uniquevals = [v for v in uniquevals if (v != exclude)]
                # if all neighbour points are excluded (hole in background?)
                if len(uniquevals) == 0:
                    logger.warn("*** all point around {} are excluded  ", point)
                    continue
    
            logger.debug('uniquevals without excluded {} ', uniquevals) 
    
            # values most present around this point? 
            count = [(neighvals==v).sum() for v in uniquevals]
            logger.debug("\t count {} ", count)
            label = uniquevals[np.argmax(count)]
    
            # a point is updated by a label
            logger.debug("\t ----- putting label {} ------------", label)
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
            logger.critical(" fill with labels : {}", filled_labels)
        else:
            logger.warn("*** ---- no progress made at this border -----  ")

    #------------ while ------------- #
    
    logger.critical("--------  exit fill_hole ----------------------- ") 
    return arr

        

