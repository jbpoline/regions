from __future__ import print_function
import os.path as osp
from warnings import warn 
import numpy as np
import nibabel as nib
from nipy.labs.datasets import VolumeImg
import numpy.testing as npt
import sys
import os 
import json
from datetime import datetime 
# from .utils import resample, atlas_mean, check_float_approximation

# this is for debug purposes
dtime = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
print("reloading ProbAtlas at %s \n" % dtime)
sys.stdout.flush()

# globals - should be a dict passed around ?
tiny = np.finfo(float).eps * 1000 # more than the usual epsilon

class ProbAtlas(object):
    # design question: should the atlas be simply a 4D nibabel image - 4th dim the roi?

    def __init__(self, prob_data, affine, labels, thres=tiny, cast_to=None):
        """
        prob_data: a 4D numpy array, 4th dimension are the regions
            (if 3D : assume that means only one region)
        affine: the associated affine
        labels: list of strings with same order as the regions
        thres: threshold above which (stricly) values are included in mask
        """
        if prob_data == None:
            self._data = None
            self.nrois = None
            self.affine = None
            self.labels = None
            return

        # affine should be 4x4
        try:
            affine.shape == (4,4)
        except ValueError as ve:
            print("affine should be 4x4, got {0}, error {1}".format(affine.shape, ve))

        # make sure affine is float
        self.affine = affine.astype('float')

        if len(prob_data.shape) == 3:   # single region in atlas
            prob_data = prob_data[..., np.newaxis]

        if cast_to != None:
            prob_data = prob_data.astype(cast_to)

        if thres != None:
            if prob_data.dtype in ['float', 'float32', 'float64']:
                prob_data[prob_data < tiny] = 0
            else:
                warn("cannot clean the data - data not float: %s " % prob_data.dtype)

        self._data = prob_data
        self.shape = prob_data.shape

        self.labels = labels
        self.nrois = len(self.labels)

        #rois = [(labels[idx], prob_data[...,idx]) 
        #                            for idx in range(len(labels))]

        # check that the number of labels is the same as the data 4th dimension 
        if len(self.labels) != prob_data.shape[3]:
            raise Exception("labels %d data %d " % len(self.labels), 
                                                prob_data.shape[3])
        

    @property
    def data(self):
        return self._data.copy() 
    
       # returns a copy of the data 
       # print("\n copying data !!! \n", file=sys.__stdout__)
       # sys.stdout.flush()

       # d = self.rois[0][1][...,np.newaxis]
       # for idx in range(len(self.rois)-1):
       #     d = np.append(d, self.rois[idx+1][1][...,np.newaxis], axis=3)

    def get_mask(self, thres=tiny, roiidx='all'):
        """
        inputs:
            roiidx :
                'all' or list of roi indices
        return:
            boolean mask 
                True if data > thres in any of the rois (the 'in' mask)
        """
        if roiidx == 'all':
            roiidx = range(self._data.shape[-1])
            # print(roiidx)
            # return (self._data > thres).any(axis=3)          
        if not hasattr(roiidx, '__iter__'):
            roiidx = [roiidx] # assumes a single roi

        return (self._data[...,roiidx] > thres).any(axis=3) #.astype('bool')


    def mask_rois(self, mask, outvalue=-1):
        """
        Puts outvalue where mask is False.

        inputs:
            mask: boulean mask of the right dimension
        """
        # force recomputing the parcellation next time it is used
        self.parcellation = None
        self._data[~mask] = outvalue

    def parcellate(self, thres=tiny, outvalue=-1):
        """ 
        Create a labelled array from the list of probabilistic rois
        thres: float
            the threshold above which (stricly) the data are considered inside
            an ROI
        """
        # This uses _data : does not duplicate the data
        self.parcellation = self._data.argmax(axis=3)

        # the inverse of the mask : outcondition
        outcondition = ~self.get_mask(thres)

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
        This takes a mask (True in the mask), and put voxels that are outside the mask
        to to "cut_value" for all ROIs except those in "do_not_cut"
        inputs:
            keep_mask: 1 or True value in mask will be kept 
            prepend_str: string
                the label to be prepend
            do_not_cut: list
                list of the index of the regions to not cut
            cut_value: float
                the value to put where keep_mask is True 
        returns:
            new_data, self.affine.copy(), new_labels
            # cut_Atlas: ProbAtlas object
            #    the same atlas with regions cut
        """
         
        if verbose:
            print(" entering keep_mask stdout", file=sys.stdout)
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
                    print(" not %d " % idx, file=sys.stdout, end='')
                    sys.stdout.flush()
                    # sys.__stdout__.flush()

                new_data[...,idx] = self._data[...,idx] 
            else:
                if verbose:
                    # debug : stdout to cell output / __stdout__ to terminal
                    print(" cut %d " % idx, file=sys.stdout, end='')
                    sys.stdout.flush()

                new_data[...,idx] = np.where(keep_mask, self._data[...,idx], cut_value)
                new_labels[idx] = prepend_str + self.labels[idx]

        # force recomputing the parcellation next time it is used
        self.parcellation = None

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
        # force recomputing the parcellation next time it is used
        self.parcellation = None

    def indexof(self, roi_names):
        if not hasattr(roi_names, '__iter__'): 
            roi_names = [roi_names]

        return [self.labels.index(roi_name) for roi_name in roi_names]

    def searchin(self, pattern, regex=False):
        "return the list of roi index in which pattern is found in labels"
        if not regex:
            return([i for i in range(self.nrois) if pattern in self.labels[i]])
        else:
            import re
            regsearch = [re.search(pattern, self.labels[i]) for i in range(self.nrois)]
            return [i for (i,s) in enumerate(regsearch) if s != None]

    def append(self, Atlas2):
        """ Append  Atlas2 to self """

        if not np.allclose(self.affine, Atlas2.affine):
            warn("affines dont match in self and Atlas2 : cannot append")
            return None

        npt.assert_equal(self.shape[:-1], Atlas2.shape[:-1])

        # check that none of the roi label in self is in Atlas2
        for roi in self.labels:
            if roi in Atlas2.labels:
                warn("roi label %s also in appended atlas " % roi)
                return None

        self.nrois = self.nrois + Atlas2.nrois
        self.labels = self.labels + Atlas2.labels
        self._data = np.concatenate((self._data, Atlas2._data),axis=-1)
        self.shape = self._data.shape
        # force recomputing the parcellation next time it is used
        self.parcellation = None

    def write_to_file(self, filename, force=False):
        """
        write a nifti image and a jason file
        """
        if osp.isfile(filename) and not force:
            warn(" %s exists - use force=True to overwrite" % filename)
            return None
        else:
            fbase, _ = osp.splitext(filename) 
            img = nib.Nifti1Image(np.asarray(self._data), self.affine)
            nib.save(img, fbase+'.nii.gz')
            with open(fbase+'.json','w') as f:
                json.dump([(str(idx), l) for (idx,l) in enumerate(self.labels)], f)


    def write_to_list_files(self, directory, prefix="", postfix="", cast2=None, force=False):
        """
        write a list of image 
        """
        def _cast2(data, dt):
            if dt == None:
                return data
            else:
                return(data.astype(dt))

        # check directory is legit 
        directory = osp.realpath(directory)
        if not os.access(directory, os.W_OK):
           raise ValueError(" cannot write to : {0}".format(directory))
        
        lablist = [  lab.strip().replace(' ','_') for lab in self.labels]
        mxlen = max([len(lab) for lab in lablist])
        strfmt = "{:_<" + str(mxlen) + "}"
        fnames = [strfmt.format(lab) for lab in lablist]

        for idx, fname in enumerate(fnames):
            fname = prefix+fname+postfix
            fbase = osp.join(directory, fname)
            if osp.isfile(fbase) and not force:
                warn("cannot overwrite with this setting, {}".format(fbase))
            else:
                img = nib.Nifti1Image(np.asarray(_cast2(self._data[...,idx],cast2)), self.affine)
                nib.save(img, fbase+'.nii.gz')


    def write_parcels(self, filename, force=False, **extra):
        """
        write a nifti image of the parcellation 
        *extra : dictionary containing named arguments to pass to parcellate
        """
        if osp.isfile(filename) and not force:
            warn(" %s exists - use force=1 to overwrite" % filename)
            return None
        else:
            fbase, _ = osp.splitext(filename) 
            if not (hasattr(self, "parcellation")) or (self.parcellation==None):
                thres = tiny; outvalue = -1 
                if extra.has_key("thres"): thres = extra["thres"]
                if extra.has_key("outvalue"): outvalue = extra["outvalue"]
                self.parcellate(thres, outvalue)

            #cast parcellation as float to be read by mricron
            img = nib.Nifti1Image(self.parcellation.astype("float"), self.affine)
            nib.save(img, fbase+'.nii')

    def summary(self):
        allidx = range(self.nrois)
        print(" nrois : %d " % self.nrois, file=sys.stdout, end='')
        print(" shape %s " % str(self._data.shape), file=sys.stdout, end='')
        print(" dtype %s " % self._data.dtype, file=sys.stdout) 
        print("\nMax values : %s " % str([self._data[...,idx].max() for idx in allidx]), 
                file=sys.stdout)
        print("\nMin values : %s " % str([self._data[...,idx].min() for idx in allidx]),
                file=sys.stdout)
        print("\nROIs number of voxels : %s " % str(self.rois_length(allidx)),
                file=sys.stdout)
        sys.stdout.flush()

    def max_rois_pos(self,roiidx): 
        """
        roiidx: list or index
            the list of indexes or the index of the roi

        returns the positions of the max value for roi in roiidx
        or one position (x,y,z) if only one roi
        """
        if not hasattr(roiidx, '__iter__'): 
            # assumes a single roi
            return np.unravel_index(np.argmax(self._data[:,:,:,roiidx]), self.shape[:-1]) 
        else:
            positions = []
            for idx in roiidx:
                pos = np.unravel_index(np.argmax(self._data[:,:,:,idx]), self.shape[:-1])
                positions.append(pos)
        
        return positions

    def rois_mean(self, roiidx, data):
        """ 
        """ 
        if not (hasattr(self, "parcellation")) or (self.parcellation==None):
            self.parcellate()
        if not hasattr(roiidx, '__iter__'):
            roiidx = [roiidx]

        parcels = self.parcellation
        return [data[parcels == idx].mean() for idx in roiidx]

    def rois_length(self, roiidx):
        """ 
        """ 
        if not (hasattr(self, "parcellation")) or (self.parcellation==None):
            self.parcellate()
        if not hasattr(roiidx, '__iter__'):
            roiidx = [roiidx]

        parcels = self.parcellation
        return [(parcels == idx).sum() for idx in roiidx]


    def find_zero_rois(self, thres=tiny):
        """ 
        returns the list of indexes of the ROIs with no values above `thres`
        """ 
        zero_idx = []
        for idx in range(self.nrois):
            if np.all(self._data[...,idx] <= tiny):
                zero_idx.append(idx)
        return zero_idx



