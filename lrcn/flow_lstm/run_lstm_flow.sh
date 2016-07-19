#!/bin/bash
TOOLS="/home/xiaoyang/caffe/build/tools"
nice -n 19 $TOOLS/caffe train -solver solver_flow_vgg16_lrcn.prototxt -weights cuhk_action_temporal_vgg_16_split1.caffemodel 2>&1 |& tee log.txt
