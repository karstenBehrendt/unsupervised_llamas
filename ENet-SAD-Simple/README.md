
## Requirements
- [Torch](http://torch.ch/docs/getting-started.html), please follow the installation instructions at [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

## Before Start

Please follow [train_final.txt](./list/train_final.txt) and [val_final.txt](./list/val_final.txt) to put LLAMAS dataset in the desired folder. We'll call the directory that you cloned ENet-SAD-Simple as `$ENet_ROOT`.

## Testing
1. Put your trained model to `./experiments/pretrained`
    ```Shell
    cd $ENet_ROOT/experiments/pretrained
    ```
   You can just train the model by yourself and have a test.

2. Run test script
    ```Shell
    cd $ENet_ROOT
    sh ./experiments/test.sh
    ```
    Testing results (probability map of lane markings) are saved in `experiments/predicts/` by default.

3. Submit the results to server 
    Please follow the instructions of LLAMAS to submit your results.
    The performance of our trained model is as follows (you can also find the result in the [official site](https://unsupervised-llamas.com/llamas/benchmark_multi)):

  0: {'auc': 0.9999917047921401,
  	'precision': 0.9973478736332135,
  	'recall': 0.9971669285812986,
  	'threshold': 0.8156624779014764},

  1: {'auc': 0.26641235441069483,
  	'precision': 0.16706499250242474,
  	'recall': 0.32421710675084453,
  	'threshold': 0.467389746285059},

  2: {'auc': 0.8964441173279966,
  	'precision': 0.6147108519863974,
  	'recall': 0.7864514156111525,
  	'threshold': 0.9866214343733576},

  3: {'auc': 0.8804212880439664,
  	'precision': 0.6146173378763694,
  	'recall': 0.7364950314056325,
  	'threshold': 0.9923073247646805},

  4: {'auc': 0.4982252894705332,
  	'precision': 0.3300198333296339,
  	'recall': 0.4491891521484192,
  	'threshold': 0.8424673897462851}

## Training
1. Download the pre-trained model
    ```Shell
    cd $ENet_ROOT/experiments/models
    ```
   Download the pre-trained model [here](https://drive.google.com/open?id=1pIMThIsGn8z8rIs6WgSNzom1H8WVvP5Q) and move it to `$ENet_ROOT/experiments/models`.
2. Training ENet-SAD-Simple model
    ```Shell
    cd $ENet_ROOT
    sh ./experiments/train.sh
    ```
    The training process should start and trained models would be saved in `experiments/models/ENet-SAD-Simple` by default.  
    Then you can test the trained model following the Testing steps above. If your model position or name is changed, remember to set them to yours accordingly.

## Citation

If you use this code, please cite the following publication:

``` 
@article{hou2019learning,
  title={Learning Lightweight Lane Detection CNNs by Self Attention Distillation},
  author={Hou, Yuenan and Ma, Zheng and Liu, Chunxiao and Loy, Chen Change},
  journal={arXiv preprint arXiv:1908.00821},
  year={2019}
}
```

