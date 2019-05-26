# tf models deeplab example

There are a few things that need to be added before the unsupervised llamas dataset can smoothly be used with the deeplab framework.

See deeplab_train.py for some more information.
Make sure to set the dataset and workspace paths in deeplab_common.py and label_scripts/dataset_constants.py

This is an example script and config. It is only supposed to be used as a reference.

First create tfrecords files using the deeplab_tfrecords.py script. Read the docstring.

```python
Before using, add:
# -----------------------------------------------------------------------
_UNSUPERVISED_LLAMAS_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 58269,
        'valid': 20844,
        'test': 20929,
    },
    num_classes=5,
    ignore_label=255,
)

_BINARY_UNSUPERVISED_LLAMAS_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 58269,
        'valid': 20844,
        'test': 20929,
    },
    num_classes=2,
    ignore_labe=l255,
)
# -----------------------------------------------------------------------
into tensorflow/models/research/deeplab/datasets/data_generator.py


Replace:
-------------------------------------------------------------------------
# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', False,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', True,
                     'Only consider logits as last layers or not.')
-------------------------------------------------------------------------

and add the datasets to the dataset information
_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'unsupervised_llamas': _UNSUPERVISED_LLAMAS_INFORMATION,
    'binary_unsupervised_llamas': _BINARY_UNSUPERVISED_LLAMAS_INFORMATION,
}

```

This sample may not be up to date with the current deeplab implementation.
