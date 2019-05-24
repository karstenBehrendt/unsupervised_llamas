"""
Random scripts that just don't fit anywhere
"""

import os


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
