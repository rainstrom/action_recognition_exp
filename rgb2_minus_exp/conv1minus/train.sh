# start from trained rgb CNN
caffe train -gpu 5 -weights caffemodels/cuhk_action_recognition_vgg_16_split1_rgb_iter_10000_acc_77.caffemodel -solver conv1minus/solver.prototxt 

