from __future__ import print_function
from numpy.testing import (assert_array_almost_equal, assert_almost_equal, assert_array_equal, assert_equal)
import numpy as np
from regions.prob_atlas import *
#import sys
#import os
import tempfile
cast_to = 'float32'
tiny = np.finfo(cast_to).eps * 1000

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
    assert_array_almost_equal(T.data, d)
    assert_array_almost_equal(T.affine, a)
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

def test_get_mask():

    d, a, l = make_data()
    T = make_fake_atlas(d, a, l)
    Tmask = T.get_mask().astype('int')
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

    assert_array_almost_equal(d[...,[1,2]], T._data[...,[0,1]])
    
    assert_equal([l[1],l[2]], T.labels)
    assert_equal(T.nrois, len(l)-1)
    assert_equal(T.shape, d.shape[:-1]+(T.nrois,))


def test_write_read():
    
    T = make_fake_atlas(*make_data())
    
    with tempfile.NamedTemporaryFile(mode='w+t') as tmpf:
        tmpfilename = tmpf.name

    T.write_to_file(tmpfilename)
    N = readAtlasFile(tmpfilename)

    # os.remove(tmpfilename)

    assert_array_almost_equal(T._data, N._data)
    assert_equal(N.labels, T.labels)
    assert_equal(T.nrois, N.nrois)
    assert_equal(T.shape, N.shape)

def test_max_rois_pos():

    # d = np.arange(4*4*3*3).reshape((4,4,3,3)) + 1.
    T = make_fake_atlas(*make_data())
    # put some maxima
    pos = [(2,3,1), (0,2,1), (1,1,2)]

    for i,p in enumerate(pos): 
        T._data[p+(i,)] = 4*4*3*3*10

    for i,p in enumerate(pos): 
        assert_equal(T.max_rois_pos(i), p) 

def test_parcellate():
    N = make_fake_atlas(*make_data())
    N._data[:,:,0,0] =  2
    N._data[:,:,2,1] =  3
    N._data[:,:,1,2] =  4

    N.parcellate()
    assert_equal(N.parcellation[0,:,0], np.zeros(4))
    assert_equal(N.parcellation[0,:,2], np.ones(4))
    assert_equal(N.parcellation[0,:,1], np.ones(4)*2)


def test_rois_mean():
    N = make_fake_atlas(*make_data())
    N._data[:,:,0,0] =  2
    N._data[:,:,2,1] =  3
    N._data[:,:,1,2] =  4
    parcels = N._data.argmax(axis=3)

    d = N._data[...,0]
    expect_roi0 = [2.0, 0.0, 0.496527777778]
    mean_roi0 = [d[parcels==i].mean() for i in range(N.nrois)]
    assert_almost_equal(mean_roi0, expect_roi0)
    assert_almost_equal(N.rois_mean([0,1,2], d), expect_roi0)

    d = N._data[...,1]
    expect_roi1 = [0.48263888888888895, 3.0, 0.50347222222222221]
    mean_roi1 = [d[parcels==i].mean() for i in range(N.nrois)]
    assert_almost_equal(mean_roi1, expect_roi1)
    assert_almost_equal(N.rois_mean([0,1,2], d), expect_roi1)

    assert_almost_equal(N.rois_mean([0,1,2], d), expect_roi1)

    d = N._data[...,2]
    expect_roi2 = [0.48958333333333326, 0.0, 4.0]
    mean_roi2 = [d[parcels==i].mean() for i in range(N.nrois)]
    assert_almost_equal(mean_roi2, expect_roi2)
    assert_almost_equal(N.rois_mean([0,1,2], d), expect_roi2)

def test_searchin():

    N = make_fake_atlas(*make_data())
    assert_equal(N.searchin("r1"), [0])
    assert_equal(N.searchin("r3"), [2])
    assert_equal(N.searchin("r[2,3]", regex=True), [1,2])
    assert_equal(N.searchin("r"), [0,1,2])
    assert_equal(N.searchin("t"), [])
    assert_equal(N.searchin("r4"), [])
    assert_equal(N.searchin('r\\d', regex=True), [0,1,2])
    assert_equal(N.searchin('r.+', regex=True), [0,1,2])

def test_zero_rois():
    (d, a, l) = make_data()
    d[:,:,:,1] = tiny/2.0
    N = make_fake_atlas(d, a, l)
    assert_equal(N.find_zero_rois(thres=tiny), [1])
    d[:,:,:,2] = tiny/2.0
    N = make_fake_atlas(d, a, l)
    assert_equal(N.find_zero_rois(thres=tiny), [1,2])
    



