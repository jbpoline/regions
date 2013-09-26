import os.path as osp
import xml.etree.ElementTree as etree

import nibabel as nb
import numpy as np

from regions import Atlas

atlas_files_path = osp.abspath(__file__)
_atlases = osp.join(osp.split(atlas_files_path)[0], 'atlases')


if not osp.isdir(_atlases): 
    print "cannot locate atlas directory %s \n" % _atlases
    exit()

def harvard_oxford_atlas(name='cort', res='2mm'):
    """ Helper function to load Harvard-Oxford atlas.

        Parameters
        ----------
        name: str
            'cort' or 'sub' for cortical or sub-cortical atlases.
        res: str
            '1mm' or '2mm' for atlas resolution.
    """
    sub = '%s/HarvardOxford-sub-prob-%s.nii.gz' % (_atlases, res)
    cort = '%s/HarvardOxford-cort-prob-%s.nii.gz' % (_atlases, res)
    cort_map = '%s/HarvardOxford-Cortical.xml' % _atlases
    sub_map = '%s/HarvardOxford-Subcortical.xml' % _atlases

    label_map = {}

    if name == 'cort':
        for label in etree.parse(cort_map).findall('.//label'):
            label_map[int(label.get('index'))] = label.text

        A = nb.load(cort)

    elif name == 'sub':
        for label in etree.parse(sub_map).findall('.//label'):
            label_map[int(label.get('index'))] = label.text

        A = nb.load(sub)

    return Atlas(A.get_data() / 100., A.get_affine(), label_map)


def mni_atlas(res='2mm'):
    """ Helper function to MNI atlas.

        Parameters
        ----------
        res: str
            '1mm' or '2mm' for atlas resolution.
    """
    path = '%s/MNI-prob-%s.nii.gz' % (_atlases, res)

    label_map = {}

    for label in etree.parse('%s/MNI.xml' % _atlases).findall('.//label'):
        label_map[int(label.get('index'))] = label.text

    A = nb.load(path)

    return Atlas(A.get_data() / 100., A.get_affine(), label_map)


def juelich_atlas(res='2mm'):
    """ Helper function to Juelich atlas.

        Parameters
        ----------
        res: str
            '1mm' or '2mm' for atlas resolution.
    """
    path = '%s/Juelich-prob-%s.nii.gz' % (_atlases, res)

    label_map = {}

    for label in etree.parse('%s/Juelich.xml' % _atlases).findall('.//label'):
        label_map[int(label.get('index'))] = label.text

    A = nb.load(path)

    return Atlas(A.get_data() / 100., A.get_affine(), label_map)
