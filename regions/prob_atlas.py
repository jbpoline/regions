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

# from .utils import resample, atlas_mean, check_float_approximation

def load_atlas(atlas_name, atlas_dir, atlas_labels=''):
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

    nrois = data.shape[3]
    print "%d rois \n" % nrois

    labels = []
    atlas_labels = osp.join(atlas_dir, atlas_labels)
    if osp.isfile(atlas_labels):
        try:
            # find the elements "label" two levels below root 
            for label in etree.parse(atlas_labels).findall('.//label'):
                labels.append((label.get('index'), label.text))
        except:
            warn("could not parse %s creating dummy labels \n" % atlas_labels)
            etree.tostring(etree.parse(atlas_labels))
    else:
        warn("no file %s - creating dummy labels \n" % atlas_labels)
        labels = [(str(i),'region'+str(i)) for i in range(nrois)]

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
        if len(self.labels) != prob_data.shape[3]:
            raise Exception("labels %d data %d " % len(self.labels), 
                                                prob_data.shape[3])

        self.affine = affine.astype('float32')

        if len(prob_data.shape) == 3:   # single region in atlas
            prob_data = prob_data[...,None]


        self.shape = prob_data.shape
        self.nrois = len(self.labels)
        
        self.rois = [(labels[idx], prob_data[...,idx]) 
                                    for idx in range(self.nrois)]

    
    @property
    def labels(self):
        return [self.rois[i][0] for i in range(len(self.rois))]

    @property
    def data(self):
        return np.hstack( tuple([roi[1] for roi in self.rois]) )

    def mask(self, thresh=0., roiidx='all'):
        """
        roiidx:
            'all' or list of roi indices
        return boulean mask : True if data > thresh
        """
        
        if roiidx == 'all':
            return (self.data > thresh).sum(3).astype('bool') 
        else:
            return (self.data[...,roiidx] > thresh).sum(3).astype('bool')

    def parcellate(self, thres = 0., outvalue=-1):
        """ 
        Create a labelled array from the list of probabilistic rois
        thres: float
            the threshold above which (stricly) the data are considered inside
        """
        # This - bad design - duplicates the data...
        self.parcellation = self.data.argmax(axis=3)

        # the inverse of the mask : outcondition
        outcondition = ~self.mask(thres)

        # put -1 outside such that 0 is a relevant label
        self.parcellation = np.where(outcondition, outvalue, self.parcellation)

    def getrois(self, idxrois):
        return self.rois(idxrois)

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
        input_image = VolumeImg(self.data, # dont need self.data[:] because data copied here
                            self.affine,
                            'arbitrary',
                            interpolation=interpolation
                            )
        resampled_img = input_image.as_volume_img(target_affine, target_shape)
        
        assert np.all(resampled_img.affine == target_affine), \
                                "resampled_img.affine != target_affine"
        
        return resampled_img.get_data() 


    def cut_right_left(self, cut_mask, do_not_cut=[]):
        """ 
        A method to cut the atlas in left and right part, assumes that x is first dimension
        inputs:
            cut_mask: 0 or False value in mask will be cut
            do_not_cut: list
                list of the index of the regions to not cut
        returns:
            lrAtlas: ProbAtlas object
                the same atlas with regions cut
        """
        dimxyzr = self.shape 

        npt.assert_equal(np.asarray(dimxyzr)[:-1], cut_mask.shape, "shape of self and mask")
        
        (dx, dy, dx, dr) = self.shape 
        dx2 = dx/2
        new_rois = []
        for idx, roi in enumerate(self.rois):
            #
            if idx not in do_not_cut:
                rdata = np.array(roi[idx][1], copy=TRUE)
                rdata[dx2,...] = 0 
                rlabel = "Right_" + roi[idx][0][1]
                new_rois.append((rlabel, rdata))
                ldata = np.array(roi[idx][1], copy=TRUE)
                ldata[dx2,...] = 0
                llabel = "Left_" + roi[idx][0][1]
                new_rois.append((llabel, ldata))
        
        return new_rois + self.rois(do_not_cut)


