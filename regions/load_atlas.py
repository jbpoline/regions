from __future__ import print_function
import os.path as osp
import xml.etree.ElementTree as etree
from warnings import warn 
import numpy as np
import nibabel as nib
import sys
import json
from datetime import datetime 
import glob
# from .utils import resample, atlas_mean, check_float_approximation

# this is for debug purposes
dtime = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
print("reloading " + __name__ + " at %s \n" % dtime)
sys.stdout.flush()

# globals - should be a dict passed around ?
cast_to = 'float32'
tiny = np.finfo(cast_to).eps * 1000

def load_aal_atlas(atlas_dir, aal_basename="ROI_MNI_V4", verbose=0):
    """
    utility function to load the AAL atlas
    returns a ProbAtlas object
    """
    
    if not osp.isdir(atlas_dir):
        raise ValueError("%s not a directory" % atlas_dir)

    aal_img_name = glob.glob(osp.join(atlas_dir, aal_basename+"*.nii"))[0]
    aal_labels_name = glob.glob(osp.join(atlas_dir, aal_basename+"*.txt"))[0]
    aalimg = nib.load(aal_img_name)
    data = aalimg.get_data()

    labels = []
    with open(aal_labels_name) as f:
        for line in f.read().splitlines():
            labels.append(line.split("\t"))
    
    # labels is now a list of ["short name", "long name", "ROI_value"]
    # [['FAG', 'Precentral_L', '2001'], ['FAD', 'Precentral_R', '2002'], ...]
    n_roi = len(labels)
    split_data = np.ndarray(aalimg.shape + (n_roi,), dtype=bool)
    split_data.fill(False)
    
    only_name_labels = []
    roi_size = []
    for idx,lab in enumerate(labels):
        only_name_labels.append(lab[1])
        split_data[...,idx] = data==int(lab[2])
        roi_size.append(split_data[...,idx].sum())
    
    return (split_data, aalimg.get_affine(), only_name_labels, roi_size)

    

def load_ho_atlas(atlas_name, atlas_dir, atlas_labels='', scalefactor=1.0, 
                verbose=0, clean=True):
    """ 
    atlas_name: str
        The name of the gz and xml file
    atlas_dir: str
        The path to the atlas files
    atlas_labels : str 
        make a file : osp.join(atlas_dir, atlas_labels)
    scalefactor: float or "max"
        multiply the block of data by scalefactor
        if "max" : multiply by 1./data.max()
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

    # cast data **** HERE WE CAST THE DATA TO cast_to ***** 
    data = data.astype(cast_to)

    if not len(data.shape) in [3,4]:
        raise Exception("dimension of data not right")
    if len(data.shape) == 3: 
        # add a dimension if only 3D : one region
        data = data[...,None]
        warn("adding an extra dimension to only 3d data")
    
    nrois = data.shape[3]

    if verbose:
        print("Atlas has %d rois \n" % nrois, file=sys.stdout)
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

    if scalefactor == 'max': scalefactor = 1./data.max()
    if not scalefactor == 1.: data *= scalefactor
    if clean:
        if data.dtype in ['float', 'float32', 'float64']:
            data[data < tiny] = 0
        else:
            warn("cannot clean the data - data not float: %s " % data.dtype)

    print( " scalefactor = %f " % scalefactor, file=sys.stdout) 
    sys.stdout.flush()
        
    return (data, affine, labels)

def load_tmp_atlas(filename):
    """
    read a nifti image and a jason file, image must be .nii or .nii.gz 
    json must be .txt or .json
    The json has this format: 
       [["0", "Left_Frontal Pole"],
        ["1", "Left_Insular Cortex"],
        ["2", "Left_Superior Frontal Gyrus"],
        ["3", "Left_Middle Frontal Gyrus"],
        ...
       ]
    input:
        filename: string 
    return:
        ProbAtlas object
    """
    fbase, ext = osp.splitext(filename)
    fimg = None
    if osp.isfile(fbase+".nii"): fimg = fbase+".nii"
    if osp.isfile(fbase+".nii.gz"): fimg = fbase+".nii.gz" 

    try:
        img = nib.load(fimg)
    except ValueError as e:
        print("error {0}, cannot find file {1} .nii or .nii.gz ".format(fbase, e.errno))

    fjson = None
    if osp.isfile(fbase+".txt"): fjson= fbase+".txt"
    if osp.isfile(fbase+".json"): fjson= fbase+".json"

    if fjson == None:
        warn("cannot find file %s .txt or .json" % fbase)
        return None

    with open(fjson) as f:
        j_labels = json.load(f)

    a_labels = [label[1] for label in j_labels]
    
    return (img.get_data(), img.get_affine(), a_labels)

