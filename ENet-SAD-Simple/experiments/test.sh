#!/usr/bin/env sh
data=./data
#rm gen/lane.t7
CUDA_VISIBLE_DEVICES="6,7" th testLane.lua \
	-model experiments/pretrained/model_new.t7 \
	-data ${data} \
	-val ./list/test.txt \
	-save experiments/predicts/ \
	-dataset laneTest \
	-shareGradInput true \
	-nThreads 8 \
	-nGPU 2 \
	-batchSize 8 \
	-smooth false 
