CUDA_VISIBLE_DEVICES=0 nice -n 19 caffe train -solver solver_rgb_vgg16_lrcn.prototxt  -snapshot snapshots_rgb_vgg16_lrcn_iter_5148.solverstate | tee log_rgb_vgg16_lrcn.prototxt

