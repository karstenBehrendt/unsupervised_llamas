"""
Scripts to create the curves between lanes. (Splines here)
"""

import cv2
import numpy
import scipy.interpolate

from unsupervised_llamas.label_scripts import label_file_scripts
from unsupervised_llamas.label_scripts import dataset_constants as dc


def _draw_points(image, points, color=(255, 0, 0)):
    for point in map(tuple, points):
        cv2.circle(image, point, 2, color, 1)


class SplineCreator():
    """
    For each lane divder
      - all lines are project
      - linearly interpolated to limit oscillations
      - interpolated by a spline
      - subsampled to receive individual pixel values

    The spline creation can be optimized!
      - Better spline parameters
      - Extend lowest marker to reach bottom of image would also help
      - Extending last marker may in some cases be interesting too
    Any help is welcome.

    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.json_content = label_file_scripts.read_json(json_path)
        self.lanes = self.json_content['lanes']
        self.lane_marker_points = {}
        self.sampled_points = {}
        self.splines = {}
        self.spline_points = {}
        self.debug_image = numpy.zeros((717, 1276, 3), dtype=numpy.uint8)

    def _sample_points(self, lane, ypp=5):
        """ Markers are given by start and endpoint. This one adds extra points
        which need to be considered for the interpolation. Otherwise the spline
        could arbitrarily oscillate between start and end of the individual markers

        Parameters
        ----------
        lane: polyline, in theory but there are artifacts which lead to inconsistencies
              in ordering. There may be parallel lines. The lines may be dashed. It's messy.
        ypp: y-pixels per point, e.g. 10 leads to a point every ten pixels

        Notes
        -----
        Especially, adding points in the lower parts of the image (high y-values) because
        the start and end points are too sparse.
        Removing upper lane markers that have starting and end points mapped into the same pixel.
        """
        # NOTE lots of potential for errors. Not checked.
        points = []
        lane_ends = []
        for marker in lane['markers']:
            points.append((marker['pixel_start']['x'], marker['pixel_start']['y']))
            lane_ends.append((marker['pixel_start']['x'], marker['pixel_start']['y']))

            height = float(marker['pixel_start']['y'] - marker['pixel_end']['y'])
            if height > ypp + 2:
                num_steps = height // ypp
                slope = (marker['pixel_end']['x'] - marker['pixel_start']['x']) / height
                step_size = (marker['pixel_start']['y'] - marker['pixel_end']['y']) / float(num_steps)
                for i in range(1, int(num_steps)):
                    x = marker['pixel_start']['x'] + slope * step_size * i
                    y = marker['pixel_start']['y'] - step_size * i
                    points.append((int(round(x)), int(round(y))))
            points.append((marker['pixel_end']['x'], marker['pixel_end']['y']))
            lane_ends.append((marker['pixel_end']['x'], marker['pixel_end']['y']))

        points = sorted(points, key=lambda x: x[1])  # sorting vertically

        unique_points = []
        unique_points.append(points[0])
        for point in points[1:]:
            if point[1] > unique_points[-1][1]:
                unique_points.append(point)

        self.lane_marker_points[lane['lane_id']] = lane_ends
        self.sampled_points[lane['lane_id']] = unique_points

        return unique_points

    def _lane_spline_fit(self, lane):
        """ Fits spline in image space for the markers of a single lane (side)

        Parameters
        ----------
        lane: dict as specified in label

        Returns
        -------
        Pixel level values for curve along the y-axis

        Notes
        -----
        This one can be drastically improved. Probably fairly easy as well.
        """
        # NOTE all variable names represent image coordinates, interpolation coordinates are swapped!
        sampled_points = self._sample_points(lane, ypp=5)
        sampled_array = numpy.asarray(sampled_points).astype(float)

        x_label = sampled_array[:, 0]
        y_label = sampled_array[:, 1]
        if y_label[-1] - y_label[0] < 40:
            raise ValueError('lane too small')
        if len(lane['markers']) == 1:
            if y_label[-1] - y_label[0] > 40:
                print('Skipping surprisingly big marker')
            raise ValueError('Only one marker, skipping this one')

        try:
            spline = scipy.interpolate.interp1d(x=y_label, y=x_label, kind='cubic',
                                                bounds_error=False, fill_value='raise')

            # t is given in actual y values
            t = [y_label[(i * len(y_label)) // 5 + len(y_label) // (2 * 5)] for i in range(5)]
            spline = scipy.interpolate.LSQUnivariateSpline(x=y_label, y=x_label, t=t)
        except ValueError as e:
            # Error handling should be improved ..
            print(e)
            raise

        y_interp = numpy.linspace(y_label[0], y_label[-1], num=y_label[-1] - y_label[0], endpoint=False)
        x_interp = spline(y_interp)
        points = numpy.round(numpy.stack((x_interp, y_interp))).astype(int)
        points = numpy.transpose(points)

        self.spline_points[lane['lane_id']] = points
        self.splines[lane['lane_id']] = spline

        return points

    def create_all_splines(self,):
        """ Creates splines for given label """
        for lane in self.lanes:
            try:
                self._lane_spline_fit(lane)
            except ValueError:
                continue  # should be lane too short, noisily mapped markers

    def _show_lanes(self, return_only=False):
        """ For debugging spline creation only """

        gray_image = label_file_scripts.read_image(self.json_path, 'gray')
        self.debug_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        self.create_all_splines()

        for lane_name, spline in self.spline_points.items():
            _draw_points(self.debug_image, spline, dc.DICT_COLORS[lane_name])

        for _, sampled_points in self.sampled_points.items():
            _draw_points(self.debug_image, sampled_points, dc.DCOLORS[1])

        for lane_name, marker_points in self.lane_marker_points.items():
            _draw_points(self.debug_image, marker_points, dc.DICT_COLORS[lane_name])

        if not return_only:
            cv2.imshow('debug image', cv2.resize(self.debug_image, (2200, 1400)))
            cv2.waitKey(10000)

        return self.debug_image


def get_horizontal_values_for_four_lanes(json_path):
    """ Gets an x value for every y coordinate for l1, l0, r0, r1

    This allows to easily train a direct curve approximation. For each value along
    the y-axis, the respective x-values can be compared, e.g. squared distance.
    Missing values are filled with -1. Missing values are values missing from the spline.
    There is no extrapolation to the image start/end (yet).
    But values are interpolated between markers. Space between dashed markers is not missing.

    Parameters
    ----------
    json_path: str
               path to label-file

    Returns
    -------
    List of [l1, l0, r0, r1], each of which represents a list of ints the length of
    the number of vertical pixels of the image

    Notes
    -----
    The points are currently based on the splines. The splines are interpolated based on the
    segmentation values. The spline interpolation has lots of room for improvement, e.g.
    the lines could be interpolated in 3D, a better approach to spline interpolation could
    be used, there is barely any error checking, sometimes the splines oscillate too much.
    This was used for a quick poly-line regression training only.
    """

    sc = SplineCreator(json_path)
    sc.create_all_splines()

    l1 = sc.spline_points.get('l1', numpy.asarray([[-1, 0]]))
    l0 = sc.spline_points.get('l0', numpy.asarray([[-1, 0]]))
    r0 = sc.spline_points.get('r0', numpy.asarray([[-1, 0]]))
    r1 = sc.spline_points.get('r1', numpy.asarray([[-1, 0]]))

    def fill_values(pixel_list, vertical_pixels=717):
        min_y = pixel_list[0, 1]
        max_y = pixel_list[-1, 1]
        min_filler = [(-1, -1)] * min_y
        max_filler = [(-1, -1)] * (vertical_pixels - max_y - 1)
        filled = min_filler + pixel_list.tolist() + max_filler
        return numpy.asarray(filled)

    lanes = [l1, l0, r0, r1]
    lanes = list(map(lambda x: fill_values(x)[:, 0], lanes))  # add missing and remove y component

    return lanes
