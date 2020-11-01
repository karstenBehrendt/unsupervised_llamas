""" Evaluation script for the CULane metric on the LLAMAS dataset.

This script will compute the F1, precision and recall metrics as described in the CULane benchmark.

The predictions format is the same one used in the CULane benchmark.
In summary, for every annotation file:
    labels/a/b/c.json
There should be a prediction file:
    predictions/a/b/c.lines.txt
Inside each .lines.txt file each line will contain a sequence of points (x, y) separated by spaces.
For more information, please see https://xingangpan.github.io/projects/CULane.html

This script uses two methods to compute the IoU: one using an image to draw the lanes (named `discrete` here) and
another one that uses shapes with the shapely library (named `continuous` here). The results achieved with the first
method are very close to the official CULane implementation. Although the second should be a more exact method and is
faster to compute, it deviates more from the official implementation. By default, the method closer to the official
metric is used.
"""

import os
import argparse
from functools import partial

import cv2
import numpy as np
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon

import unsupervised_llamas.common.helper_scripts as helper_scripts
import unsupervised_llamas.label_scripts.spline_creator as spline_creator

LLAMAS_IMG_RES = (717, 1276)


def add_ys(xs, vertical_cutoff=300):
    assert len(xs) >= 717 - vertical_cutoff, "Need at least 417 pixels per lane"

    # Reference lanes go from 0 to 717. If a horizontal entry is not
    # defined, it is stored as -1. We have to filter for that.

    if len(xs) == 717:  # lane regressed across complete image
        xs = xs[vertical_cutoff:]
    elif len(xs) == 417:  # lane regress across part of image that is relevant
        pass
    else:
        raise NotImplementedError(f"Evaluations not implemented for length of detected lane: {len(xs)}")
    xs = np.array(xs)
    valid = xs >= 0
    xs = xs[valid]
    assert len(xs) > 1
    ys = np.arange(300, 717)[valid]
    return list(zip(xs, ys))


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(1,), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=LLAMAS_IMG_RES):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=LLAMAS_IMG_RES):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in xs]
    ys = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., n)
    return np.array(splev(u, tck)).T


def culane_metric(pred, anno, width=30, iou_threshold=0.5, unofficial=False, img_shape=LLAMAS_IMG_RES):
    if len(pred) == 0:
        return 0, 0, len(anno)
    if len(anno) == 0:
        return 0, len(pred), 0
    interp_pred = np.array([interp(pred_lane, n=50) for pred_lane in pred])  # (4, 50, 2)
    interp_anno = np.array([interp(anno_lane, n=50) for anno_lane in anno])  # (4, 50, 2)

    if unofficial:
        ious = continuous_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    else:
        ious = discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp
    fn = len(anno) - tp
    # pred_ious = np.zeros(len(pred))
    # pred_ious[row_ind] = ious[row_ind, col_ind]
    return tp, fp, fn


def load_prediction(path):
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def load_prediction_list(label_paths, pred_dir):
    return [load_prediction(os.path.join(pred_dir, path.replace('.json', '.lines.txt'))) for path in label_paths]


def load_labels(label_dir):
    label_paths = helper_scripts.get_files_from_folder(label_dir, '.json')
    annos = [[add_ys(xs) for xs in spline_creator.get_horizontal_values_for_four_lanes(label_path) if
              (np.array(xs) >= 0).sum() > 1]  # lanes annotated with a single point are ignored
             for label_path in label_paths]
    label_paths = [
        helper_scripts.get_label_base(p) for p in label_paths
    ]
    return np.array(annos, dtype=object), np.array(label_paths, dtype=object)


def eval_predictions(pred_dir, anno_dir, width=30, unofficial=True, sequential=False):
    print('Loading annotation data ({})...'.format(anno_dir))
    annotations, label_paths = load_labels(anno_dir)
    print('Loading prediction data ({})...'.format(pred_dir))
    predictions = load_prediction_list(label_paths, pred_dir)
    print('Calculating metric {}...'.format('sequentially' if sequential else 'in parallel'))
    if sequential:
        results = t_map(partial(culane_metric, width=width, unofficial=unofficial, img_shape=LLAMAS_IMG_RES), predictions,
                        annotations)
    else:
        results = p_map(partial(culane_metric, width=width, unofficial=unofficial, img_shape=LLAMAS_IMG_RES), predictions,
                        annotations)
    total_tp = sum(tp for tp, _, _ in results)
    total_fp = sum(fp for _, fp, _ in results)
    total_fn = sum(fn for _, _, fn in results)
    if total_tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = float(total_tp) / (total_tp + total_fp)
        recall = float(total_tp) / (total_tp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)

    return {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Precision': precision, 'Recall': recall, 'F1': f1}


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric on the LLAMAS dataset")
    parser.add_argument("--pred_dir", help="Path to directory containing the predicted lanes", required=True)
    parser.add_argument("--anno_dir", help="Path to directory containing the annotated lanes", required=True)
    parser.add_argument("--width", type=int, default=30, help="Width of the lane")
    parser.add_argument("--sequential", action='store_true', help="Run sequentially instead of in parallel")
    parser.add_argument("--unofficial", action='store_true', help="Use a faster but unofficial algorithm")

    return parser.parse_args()


def main():
    args = parse_args()
    results = eval_predictions(args.pred_dir,
                               args.anno_dir,
                               width=args.width,
                               unofficial=args.unofficial,
                               sequential=args.sequential)

    header = '=' * 20 + ' Results' + '=' * 20
    print(header)
    for metric, value in results.items():
        if isinstance(value, float):
            print('{}: {:.4f}'.format(metric, value))
        else:
            print('{}: {}'.format(metric, value))
    print('=' * len(header))


if __name__ == '__main__':
    main()
