from __future__ import print_function
from numpy.testing import (assert_array_almost_equal, assert_almost_equal, assert_array_equal, assert_equal)
import numpy as np
from regions.prob_atlas import *
import sys

def make_data():
    d = np.arange(4*4*3*3).reshape((4,4,3,3)) + 1.
    d = d/d.max() # d values between 0 and 1.

    # make the last "slice" zero:
    d[:,:,2,:] = 0
    a = np.eye(4)
    # l = [('1','r1'),('2','r2'),('3','r3')]
    l = ['r1','r2','r3']
    return (d, a, l) 

def make_fake_atlas(d, a, l):
    # a fake little one
    return ProbAtlas(d,a,l)

def test_atlas():
    
    d, a, l = make_data()
    T = make_fake_atlas(d, a, l)
    assert T.nrois == 3
    # assert_array_equal(np.asarray(T.labels), np.asarray(['r1', 'r2', 'r3']))
    assert_equal(T.labels,['r1', 'r2', 'r3'], " T.labels and ['r1', 'r2', 'r3']")
    assert_equal(T.shape, d.shape)
    assert_array_equal(T.data, d)
    assert_array_equal(T.affine, a)
    assert T.labels == l

def test_keep_mask():
    
    d, a, l = make_data()
    msk = d[...,0]
    msk[0:2] = 0
    msk[2:] = 1

    prependStr = 'Right_'
    T = make_fake_atlas(d, a, l)
    (kd, ka, kl) = T.keep_mask(msk, prependStr, [1])

    # make sure the original affine is returned
    assert_array_almost_equal(ka, T.affine)
    
    # make sure the data is well masked 
    assert_array_almost_equal(kd[...,0], msk*T.data[...,0])
    assert kl[0] == prependStr + T.labels[0]  

    assert_array_almost_equal(kd[...,1], T.data[...,1])
    assert kl[1] == T.labels[1]  

def test_mask():

    d, a, l = make_data()
    T = make_fake_atlas(d, a, l)
    Tmask = T.mask().astype('int')
    # that doesnt work :
        # print Tmask
        # sys.stdout.flush()
    # that works :
    #print(" printing mask: \n%s " % np.array2string(Tmask), file=sys.__stdout__)
    #sys.__stdout__.flush()

    # the slice zeroed
    zeroed_slice = 2
    assert_array_almost_equal(Tmask[:,:,zeroed_slice], np.zeros_like(T._data[:,:,0,0]))
    assert_array_almost_equal(Tmask[:,:,1], np.ones_like(T._data[:,:,0,0]))

def test_resample():

    d, a, l = make_data()
    T = make_fake_atlas(d, a, l)

    # make it twice bigger:
    new_a = a.copy()
    new_a[[0,1,2],[0,1,2]] = .5 #fancy indexing
    new_shape = np.asarray(d.shape) * 2
    new_shape[-1] = T.shape[-1]

    target = (new_shape[:-1], new_a)
    new_d = T.resample(target, "continuous")
    #new_d = T.resample(target, "nearest")

    # resample towards bigger volume
    assert_equal(new_d.shape, new_shape)
    
    # resample back:
    N = ProbAtlas(new_d, new_a, l)
    new_new_d = N.resample((T.shape[:-1], a), "nearest")

    assert_array_almost_equal(new_new_d, d)

def test_rm_rois():

    d, a, l = make_data()
    T = make_fake_atlas(d, a, l)
    T.rm_rois([0])

    assert_array_equal(d[...,[1,2]], T._data[...,[0,1]])
    
    assert_equal([l[1],l[2]], T.labels)
    assert_equal(T.nrois, len(l)-1)
    assert_equal(T.shape, d.shape[:-1]+(T.nrois,))







