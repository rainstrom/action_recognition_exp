import caffe
import numpy as np
caffe.set_cpu_mode()

net = caffe.Net("train_test_rgb_vgg16_lrcn.prototxt", "snapshots_rgb_vgg16_lrcn_iter_5000.caffemodel", caffe.TEST)
total_num = 3784
correct_num = 0.0
mean_correct_num = 0.0
final_correct_num = 0.0
for i in range(total_num):
    net.forward()
    fc8 = net.blobs["fc8-lrcn"].data
    label = net.blobs["label"].data

    # prediction of all time_step(0), batch(1), 101
    pd = np.argmax(fc8, axis=2)
    correct_num += np.mean(pd == label, dtype=np.float32)

    pd_final = np.argmax(fc8[-1,:,:], axis=1)
    final_correct_num += np.mean(pd_final == label[-1,:], dtype=np.float32)

    pd_mean = np.argmax(np.mean(fc8, axis=0), axis=1)
    mean_correct_num += np.mean(pd_mean == label[-1,:], dtype=np.float32)

    running_num = i + 1
    print "running ", running_num
    print "general correct rate: ", correct_num / running_num
    print "mean correct rate: ", mean_correct_num / running_num
    print "final correct rate: ", final_correct_num / running_num
