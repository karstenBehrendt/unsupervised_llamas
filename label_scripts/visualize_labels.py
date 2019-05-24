"""
A collection of functions to draw the labels
"""

import cv2
import numpy

from unsupervised_llamas.label_scripts.spline_creator import SplineCreator
from unsupervised_llamas.label_scripts import label_file_scripts
from unsupervised_llamas.label_scripts import dataset_constants as dc


def _draw_lines(debug_image, spline, color):
    # NOTE spline given as points
    last_point = None
    for i, point in enumerate(map(tuple, spline)):
        if i == 0:
            last_point = point
            continue
        try:  # difference in cv2 versions
            aliasing = cv2.LINE_AA
        except AttributeError:
            aliasing = cv2.CV_AA
        cv2.line(debug_image, point, last_point, color, 3, aliasing)
        last_point = point


def create_spline_image(json_path, image='blank'):
    """ Draws splines into given image

    Parameters
    ----------
    json_path: str
               path to label file
    image: str, 'blank' for all zeros or 'gray' for gray image
           numpy.array, direct image input

    Returns
    -------
    numpy.array
        image with drawn splines
    """
    sc = SplineCreator(json_path)
    sc.create_all_splines()

    # TODO replace section by label_file_scripts read_image
    if isinstance(image, str):
        if image == 'blank':
            image = numpy.zeros((717, 1276, 3), dtype=numpy.uint8)
        elif image == 'gray':
            image = label_file_scripts.read_image(json_path, 'gray')
        else:
            raise ValueError('Unexpected input image: {}'.format(image))

    # TODO Request that as part of read_image as well or util function
    if (len(image.shape) == 2 or image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for lane_name, spline in sc.spline_points.items():
        _draw_lines(image, spline, dc.DICT_COLORS[lane_name])

    return image


def create_segmentation_image(json_path, color=None, image=None):
    """ Draws pixel-level markers onto image

    Parameters
    ----------
    json_path: str
               path to label-file
    color: int/uint8 for grayscale color to draw markers
           tuple (uint8, uint8, uint8), BGR values
           None for default marker colors, multi-class
    image: str, 'blank' for all zeros or 'gray' for gray image
           numpy.array, direct image input

    Returns:
    --------
    numpy.array
        image with drawn markers

    Notes
    -----
    This one is for visualizing the label, may not be optimal for training label creation
    """

    label = label_file_scripts.read_json(json_path)

    # TODO replace section by label_file_scripts read_image
    # NOTE Same in function above
    if isinstance(image, str):
        if image == 'blank':
            image = numpy.zeros((717, 1276), dtype=numpy.uint8)
        elif image == 'gray':
            image = label_file_scripts.read_image(json_path, 'gray')
        # TODO Add color
        else:
            raise ValueError('Unknown image type {}'.format(image))

    if (len(image.shape) == 2 or image.shape[2] == 1)\
            and (color is None or not isinstance(color, int)):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for lane in label['lanes']:
        lane_id = lane['lane_id']
        for marker in lane['markers']:
            p1 = marker['world_start']
            p1 = [p1['x'], p1['y'], p1['z']]
            p2 = marker['world_end']
            p2 = [p2['x'], p2['y'], p2['z']]
            dcolor = dc.DICT_COLORS[lane_id] if color is None else color
            label_file_scripts.project_lane_marker(
                p1, p2, width=.1, projection_matrix=label['projection_matrix'],
                color=dcolor, img=image)
    return image
