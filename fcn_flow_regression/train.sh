caffe train -solver ./fcn_solver.prototxt -weights ./cuhk_action_recognition_vgg_16_split1_rgb_iter_10000_acc_77.caffemodel -gpu 6 | tee log_fcn_flow.txt
