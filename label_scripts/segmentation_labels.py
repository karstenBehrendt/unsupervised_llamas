"""
Collection of functions to create segmentation labels
"""

import numpy

from unsupervised_llamas.label_scripts import visualize_labels
from unsupervised_llamas.label_scripts import dataset_constants as dc


def create_multi_class_segmentation_label(json_path):
    """ Creates pixel-level label of markings color coded by lane association
    Only for the for closest lane dividers, i.e. l1, l0, r0, r1

    Parameters
    ----------
    json_path: str
               path to label file

    Returns
    -------
    numpy.array
        pixel level segmentation with lane association (717, 1276, 5)

    Notes
    -----
    Only draws 4 classes, can easily be extended for to a given number of lanes
    """
    debug_image = visualize_labels.create_segmentation_image(json_path, image='blank')

    l1 = (debug_image == dc.DICT_COLORS['l1']).all(axis=2).astype(numpy.uint8)
    l0 = (debug_image == dc.DICT_COLORS['l0']).all(axis=2).astype(numpy.uint8)
    r0 = (debug_image == dc.DICT_COLORS['r0']).all(axis=2).astype(numpy.uint8)
    r1 = (debug_image == dc.DICT_COLORS['r1']).all(axis=2).astype(numpy.uint8)

    no_marker = (l1 + l0 + r0 + r1) == 0

    return numpy.stack((no_marker, l1, l0, r0, r1), axis=2)


def create_binary_segmentation_label(json_path):
    """ Creates binary segmentation image from label

    Parameters
    ----------
    json_path: str
               path to label file

    Returns
    -------
    numpy.array
        binary image, 0 for background or 1 for marker, (716, 1276), numpy.uint8
    """
    blank_image = numpy.zeros((717, 1276), dtype=numpy.uint8)
    blank_image = visualize_labels.create_segmentation_image(
        json_path, color=1, image=blank_image)

    return blank_image
