import numpy as np
import glob

netproto = "only_test_rgb_vgg16_lrcn.prototxt"
snapshots = "snapshots_rgb_vgg16_lrcn_iter_[0-9]*.caffemodel"
total_num = 3784
total_steps = 15

snapshots_files = glob.glob(snapshots)
iternums = [int(fn.split('.')[0].split('_')[-1]) for fn in snapshots_files]
snapshot = snapshots_files[iternums.index(max(iternums))]
print("chossing caffemodel %s" % snapshot)

import caffe
caffe.set_mode_gpu()

net = caffe.Net(netproto, snapshot, caffe.TEST)
global_acc = 0.0
mean_acc = 0.0
final_acc = 0.0

def test_one_sample(fc8, labels):
    assert all([labels[0]==label for label in labels])
    label = int(labels[0])
    global_correct_num = np.sum(np.argmax(fc8, axis=1) == labels)
    mean_correct = label == np.argmax(np.mean(fc8, axis=0))
    final_correct = label == np.argmax(fc8[-1, :])
    return global_correct_num, mean_correct, final_correct

for i in range(total_num):
    tested_samples = i + 1
    net.forward()
    # time_step, batch_size, 101
    fc8 = net.blobs["fc8-final"].data
    # time_step, batch_size
    labels = net.blobs["label"].data

    global_correct_num, mean_correct, final_correct = test_one_sample(fc8[:,0,:], labels[:,0])
    global_acc += global_correct_num
    mean_acc += mean_correct
    final_acc += final_correct
    
    print("step %d", tested_samples)
    print("global acc: %.4f %d/%d" % (global_acc / (tested_samples * total_steps), global_acc, tested_samples * total_steps))
    print("mean acc: %.4f %d/%d" % (mean_acc / tested_samples, mean_acc, tested_samples))
    print("final frame acc: %.4f %d/%d" % (final_acc / tested_samples, final_acc, tested_samples))
