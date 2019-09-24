""" Scripts to check labels

Will be extended with each bug report

Usage:
    python check_labels.py some_label_folder
"""
import json
import sys

import tqdm

from unsupervised_llamas.common import helper_scripts


def check_labels(input_folder):
    """ Checks if labels within folder are readable """
    label_files = helper_scripts.get_files_from_folder(input_folder, 'json')
    for label_file in tqdm.tqdm(label_files, desc='checking labels'):
        with open(label_file, 'r') as lf:
            json.load(lf)  # Just to check if json syntax is correct


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit()
    check_labels(sys.argv[1])
