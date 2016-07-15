CUDA_VISIBLE_DEVICES=1 nice -n 19 caffe train -solver solver_flow_vgg16_lrcn.prototxt -weights VGG_ILSVRC_16_layers_forflow.caffemodel | tee log_flow_vgg_lrcn.txt
