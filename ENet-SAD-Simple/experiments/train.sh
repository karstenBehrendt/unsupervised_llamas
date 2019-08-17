#!/usr/bin/env sh
data=./data
CUDA_VISIBLE_DEVICES="4,5,6,7" th main.lua \
   -data ${data} \
   -train ./list/train_final.txt \
   -val ./list/val_final.txt \
   -dataset lane \
   -save experiments/models/ENet-SAD-Simple \
   -retrain experiments/models/ENet-label-new.t7 \
   -shareGradInput true \
   -nThreads 8 \
   -nGPU 4 \
   -batchSize 12 \
   -maxIter 100000 \
   -LR 0.01 \
   -backWeight 0.4 \
   -nEpochs 100 \
2>&1|tee experiments/models/train.log
