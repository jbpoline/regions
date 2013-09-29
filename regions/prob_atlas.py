from __future__ import print_function
from collections import namedtuple
import os.path as osp
import xml.etree.ElementTree as etree
from warnings import warn 
import numpy as np
import pylab as pl
import nibabel as nib
import nipy.labs.viz as viz
from nipy.labs.datasets import VolumeImg
import numpy.testing as npt
import sys

# from .utils import resample, atlas_mean, check_float_approximation

print("reloading ProbAtlas\n")
sys.stdout.flush()

def load_atlas(atlas_name, atlas_dir, atlas_labels='', verbose=0):
    """ 
    atlas_name: str
        The name of the gz and xml file
    atlas_dir: str
        The path to the atlas files

    Returns:
    
    data: numpy array
        the atlas 4d volume
    affine: numpy array
        the atlas affine
    labels: list
        (str1, str2)
        

    Example : (data, affine, labels) = load_atlas("HarvardOxford-cort-prob-2mm.nii.gz",
                                          osp.join(USER,'code','regions','regions','atlases'))
    """
    
    atlas_file = osp.join(atlas_dir, atlas_name)

    if not osp.isfile(atlas_file):
        raise Exception("%s not a file" % atlas_file)
    
    try:
        img = nib.load(atlas_file)
        data = img.get_data()
        affine = img.get_affine()
    except:
        raise Exception(" cannot load image %s with nibabel" % atlas_file)

    if not len(data.shape) in [3,4]:
        raise Exception("dimension of data not right")
    if len(data.shape) == 3: 
        # add a dimension if only 3D : one region
        data = data[...,None]
        warn("adding an extra dimension to only 3d data")

    if verbose:
        nrois = data.shape[3]
        print("Atlas has %d rois \n" % nrois, file=sys.__stdout__)
        # print "\nMax values : ", [data[...,idx].max() for idx in range(nrois)]
        # print "\nMin values : ", [data[...,idx].min() for idx in range(nrois)]

    labels = []
    atlas_labels = osp.join(atlas_dir, atlas_labels)
    if osp.isfile(atlas_labels):
        try:
            # find the elements "label" two levels below root 
            for label in etree.parse(atlas_labels).findall('.//label'):
                labels.append((label.get('index'), label.text))
            labels = [lab[1] for lab in labels]
        except:
            warn("could not parse %s creating dummy labels \n" % atlas_labels)
            etree.tostring(etree.parse(atlas_labels))
    else:
        warn("no file %s - creating dummy labels \n" % atlas_labels)
        # labels = [(str(i),'region'+str(i)) for i in range(nrois)]
        labels = ['region'+str(i) for i in range(nrois)]

    if not len(labels) == data.shape[3]:
        raise Exception(" %d labels and %d images in atlas \n" % (len(labels), data.shape[3]))

    return (data, affine, labels)

class ProbAtlas(object):
    # design issues : should the atlas be simply a 4D nibabel image - 4th dim the roi?

    def __init__(self, prob_data, affine, labels, incl_thres=0.):
        """
        prob_data: a 4D numpy array, 4th dimension are the regions
            (if 3D : assume that means only one region)
        affine: the associated affine
        labels: list of strings with same order as the regions
        incl_thres: threshold above which stricly values are included in mask
        """

        # check that the number of labels is the same as the data 4th dimension 

        self.affine = affine.astype('float32')

        if len(prob_data.shape) == 3:   # single region in atlas
            prob_data = prob_data[...,None]

        self._data = prob_data
        self.shape = prob_data.shape

        self.labels = labels
        self.nrois = len(self.labels)

        #rois = [(labels[idx], prob_data[...,idx]) 
        #                            for idx in range(len(labels))]

        if len(self.labels) != prob_data.shape[3]:
            raise Exception("labels %d data %d " % len(self.labels), 
                                                prob_data.shape[3])
    
    @property
    def data(self):
        return self._data.copy() 
    
       # return a copy of the data 
       # print("\n copying data !!! \n", file=sys.__stdout__)
       # sys.stdout.flush()

       # d = self.rois[0][1][...,np.newaxis]
       # for idx in range(len(self.rois)-1):
       #     d = np.append(d, self.rois[idx+1][1][...,np.newaxis], axis=3)

    def mask(self, thresh=0., roiidx='all'):
        """
        inputs:
            roiidx :
                'all' or list of roi indices
        return:
            boulean mask 
                True if data > thresh in any of the rois (the 'in' mask)
        """
        
        if roiidx == 'all':
            return (self._data > thresh).any(axis=3) #         
        else:
            if not hasattr(roiidx, '__iter__'):
                # assumes a single roi
                roiidx = [roiidx]
            return (self._data[...,roiidx] > thresh).any(axis=3) #.astype('bool')

    def parcellate(self, thres=0., outvalue=-1):
        """ 
        Create a labelled array from the list of probabilistic rois
        thres: float
            the threshold above which (stricly) the data are considered inside
        """
        # This - bad design - duplicates the data...
        self.parcellation = self._data.argmax(axis=3)

        # the inverse of the mask : outcondition
        outcondition = ~self.mask(thres)

        # put -1 outside such that 0 is a relevant label
        self.parcellation = np.where(outcondition, outvalue, self.parcellation)

#    def getrois(self, idxrois):
#        return self.rois(idxrois)
#

    def resample(self, target, interpolation='continuous'):
        """ resample to target - st after 
        inputs:
            target: (shape, affine) tuple 
                 data: numpy array, affine: numpy array
            interpolation: 'nearest' or 'continuous'
        returns:
            interpolated_data
        """
        target_shape, target_affine = target
        input_image = VolumeImg(self._data[:], #why do we need self.data[:]? 
                            self.affine,
                            'arbitrary',
                            interpolation=interpolation
                            )
        resampled_img = input_image.as_volume_img(target_affine, target_shape)
        
        assert np.all(resampled_img.affine == target_affine), \
                                "resampled_img.affine != target_affine"
        
        return resampled_img.get_data() 


    def keep_mask(self, keep_mask, prepend_str, do_not_cut=[], cut_value=0., verbose=0):
        """ 
        A method to cut the atlas in left and right part, assumes that x is first dimension
        inputs:
            keep_mask: 1 or True value in mask will be kept 
            prepend_str: string
                the label to be prepend
            do_not_cut: list
                list of the index of the regions to not cut
            cut_value: float
                the value put where keep_mask is True 
        returns:
            cut_Atlas: ProbAtlas object
                the same atlas with regions cut
        """
         
        if verbose:
            print( " entering keep_mask stdout", file=sys.stdout)
            sys.stdout.flush()

        dimxyzr = self.shape 

        npt.assert_equal(np.asarray(dimxyzr)[:-1], 
                         np.asarray(keep_mask.shape), "shape of self and mask")

        new_labels = self.labels[:]
        new_data = np.empty_like(self._data) 

        for idx in range(self.nrois):
            if idx in do_not_cut:
                if verbose:
                    # debug : stdout to cell output / __stdout__ to terminal
                    # print( " do not cut %d " % idx, file=sys.__stdout__)
                    print(" stdout do not cut %d " % idx, file=sys.stdout)
                    sys.stdout.flush()
                    # sys.__stdout__.flush()

                new_data[...,idx] = self._data[...,idx] 
            else:
                if verbose:
                    # debug : stdout to cell output / __stdout__ to terminal
                    print(" stdout do not cut %d " % idx, file=sys.stdout)
                    sys.stdout.flush()

                new_data[...,idx] = np.where(keep_mask, self._data[...,idx], cut_value)
                new_labels[idx] = prepend_str + self.labels[idx]
            
        return (new_data, self.affine.copy(), new_labels) 

    #def xyz_argmax(self, roiidx):

    def rm_rois(self, roiidx, verbose=0): 
        
        if not hasattr(roiidx, '__iter__'):
            roiidx = [roiidx]

        keepidx = [idx for idx in range(self.nrois) if idx not in roiidx]
        new_nrois = len(keepidx)

        self._data = self._data[...,keepidx]
        self.labels = [self.labels[idx] for idx in keepidx]
        self.shape = self._data.shape
        self.nrois = new_nrois

    def indexof(self, roi_names):
        if not hasattr(roi_names, '__iter__'): 
            roi_names = [roi_names]

        return [self.labels.index(roi_name) for roi_name in roi_names]


    def append(self, Atlas2):
        # Append Atlas2 to self
        if not np.allclose(self.affine, Atlas2.affine):
            warn("affines dont match in self and Atlas2")
            return None

        assert_equal(self.shape[:-1], Atlas2.shape[:-1])

        for roi in self.labels:
            if roi in Atlas.labels:
                warn("roi label %s also in appended atlas " % roi)
                return None

        self.nrois = self.nrois + Atlas2.nrois
        self.labels = self.labels + Atlas2.labels
        self._data = np.hstack(self._data, Atlas._data)






