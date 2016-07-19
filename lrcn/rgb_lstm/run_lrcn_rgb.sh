TOOLS="/home/xiaoyang/caffe/build/tools"
#nice -n 19 $TOOLS/caffe train -solver solver_rgb_vgg16_lrcn.prototxt -weights /home/xiaoyang/caffemodels/cuhk_action_spatial_vgg_16_split1.caffemodel 2>&1 |& tee log.txt
nice -n 19 $TOOLS/caffe train -solver solver_rgb_vgg16_lrcn.prototxt -snapshot snapshots_rgb_vgg16_lrcn_iter_1000.solverstate 2>&1 |& tee log.txt
