"""
Random scripts that just don't fit anywhere
"""

import os
from unsupervised_llamas.label_scripts import dataset_constants


def get_labels(split='test'):
    """ Gets label files of specified dataset split """
    label_paths = get_files_from_folder(
        os.path.join(dataset_constants.LABELS, split), '.json')
    return label_paths


def get_files_from_folder(directory, extension=None):
    """Get all files within a folder that fit the extension """
    # NOTE Can be replaced by glob for new python versions
    label_files = []
    for root, _, files in os.walk(directory):
        for some_file in files:
            label_files.append(os.path.abspath(os.path.join(root, some_file)))
    if extension is not None:
        label_files = list(filter(lambda x: x.endswith(extension), label_files))
    return label_files


def get_label_base(label_path):
    """ Gets directory independent label path """
    return '/'.join(label_path.split('/')[-2:])
