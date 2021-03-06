from __future__ import print_function
from numpy.testing import  assert_array_equal
from nose.tools import assert_raises
     # (assert_array_almost_equal, assert_almost_equal, assert_equal)
import numpy as np
from regions.clean_atlas import fill_hole_mult_label, neigh_coords, point_is_in_coords, \
                                neigh_values, get_border_coords, translate_struct
#import sys
#import os
# import tempfile
import scipy.ndimage.morphology as morph
#import scipy.ndimage as ndi 

boule = morph.generate_binary_structure(3,1)

#  In [18]: boule
#  Out[18]: 
#  array([[[False, False, False],
#          [False,  True, False],
#          [False, False, False]],
#  
#         [[False,  True, False],
#          [ True,  True,  True],
#          [False,  True, False]],
#  
#         [[False, False, False],
#          [False,  True, False],
#          [False, False, False]]], dtype=bool)

def make_testarr(background=-1, hole=0):
    testarr = np.zeros((5,5,7), dtype=int)
    testarr.fill(background)
    testarr[1:-1,1:3,1:-1] = 1
    testarr[1:-1,3:5,1:-1] = 2
    testarr[2:4,2:4,2:4] = hole
    return testarr 



def test_translate_struct():
    point = [0,10,10]
    boule_trans = translate_struct(point, boule)
    result = np.asarray([[-1, 0,  0,  0,  0,  0,  1], 
                         [10,  9, 10, 10, 10, 11, 10], 
                         [10, 10,  9, 10, 11, 10, 10]] )
    
    assert_array_equal(boule_trans, result)
    assert_raises(TypeError, translate_struct, [0,10,10,20], boule)

def test_neigh_coords():
    dim = [11,11,11]
    point = [0,10,10]
    res1 = np.asarray([[0, 0, 0, 1],[ 9,10,10,10],[10, 9,10,10]])
    res2 = neigh_coords(point, boule, dim)
    assert_array_equal(res1, res2)

    dim = [3,3,3]
    point = [1,1,1]
    res1 = np.asarray([[0,1,1,1,1,1,2],[1,0,1,1,1,2,1],[1,1,0,1,2,1,1]])
    res2 = neigh_coords([1,1,1], boule, dim)
    assert_array_equal(res1, res2)

def test_neigh_values():
    """
    """
    testarr = make_testarr(background=-1, hole=0)
    point = [2,2,2]
    res1 = np.asarray([1, 1, 1, 0, 0, 0, 0])
    res2 = neigh_values(point, boule, testarr)
    assert_array_equal(res1, res2)
    
    point = [1,1,1]
    res1 = np.asarray([-1, -1, -1, 1, 1, 1, 1])
    res2 = neigh_values(point, boule, testarr)
    assert_array_equal(res1, res2)

    point = [0,0,0]
    res1 = np.asarray([-1, -1, -1, -1])
    res2 = neigh_values(point, boule, testarr)
    assert_array_equal(res1, res2)


def test_get_border_coords():
    """
    """
    testarr = np.ones((3,3,3))
    
    border_coords = get_border_coords(testarr, boule)
    # border equals region here
    assert point_is_in_coords([0,0,0], border_coords)
    assert point_is_in_coords([0,1,0], border_coords)
    assert point_is_in_coords([1,0,0], border_coords)
    assert ~point_is_in_coords([1,1,1], border_coords)

    testarr = np.ones((3,3,3))
    testarr[1,1,1] = 2

    border_coords = get_border_coords(testarr, boule, label=2)
    assert point_is_in_coords([1,1,1], border_coords)
    assert ~point_is_in_coords([0,1,0], border_coords)
    assert ~point_is_in_coords([1,0,0], border_coords)
    assert ~point_is_in_coords([0,1,1], border_coords)

def make_testarr2():
    testarr = np.zeros((5,5,7), dtype=int)
    testarr.fill(-1)
    testarr[1:-1,1:3,1:-1] = 1
    testarr[1:-1,3:5,1:-1] = 2
    testarr[1:4,1:4,1:4] = 0
    return testarr

def test_fill_hole_mult_labels():
    """
    """
    testarr = make_testarr2()
    #coords2fill = np.where(testarr == 0)
    #border = get_border_coords(testarr, boule, label=0)
    
    fillarr = fill_hole_mult_label(testarr, testarr==0, boule, exclude=-1)

    pt = [1,3,1] # should be 2
    assert fillarr[tuple(pt)] == 2
    pt = [1,1,3] # should be 1
    assert fillarr[tuple(pt)] == 1

def test_fill_hole_mult_label():
    """
    """
    testarr = make_testarr2()
    #print(testarr)
    boule1 = morph.generate_binary_structure(3,1)
    filledarr = fill_hole_mult_label(testarr, testarr==0, boule1, exclude=-1,
                                    loglevel=10);

    assert_array_equal(filledarr[1,1,:], [-1,  1,  1,  1,  1,  1, -1])
    assert_array_equal(filledarr[1,2,:], [-1,  2,  1,  1,  1,  1, -1])
    assert_array_equal(filledarr[2,2,:], [-1,  2,  1,  1,  1,  1, -1])

    filledarr = fill_hole_mult_label(testarr, testarr==0, boule1, exclude=None,
                                    loglevel=10);

    assert_array_equal(filledarr[1,1,:], [-1, -1,  -1, -1, 1, 1, -1])
    assert_array_equal(filledarr[1,2,:], [-1, -1,  -1, -1, 1, 1, -1])
    assert_array_equal(filledarr[2,3,:], [-1, -1,  2,  2,  2,  2, -1])


    
