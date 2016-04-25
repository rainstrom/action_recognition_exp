# tags = ["rgb_original_mirror_crop", \
# 	"flow_original_mirror_crop", \
# 	"conv1_1_minus_rgb5_mirror_crop"]

# fpkls= ["./full_test_rgb_fc8_mirror_crop_seg25.pkl",\
# 	"./full_test_flow_fc8_mirror_crop_seg25.pkl",
# 	"full_test_conv1_1minus_rgb5_initfromvgg_iter30000_mirror_crop_seg25.pkl"]

# score_weights = [[1, 2, 1], [1, 2, 2], [1, 2, 3], [1, 2, 4], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 0, 1], [1, 0, 2], [2, 0, 1], [1, 1, 0], [1, 2, 0], [2, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 1], [1, 6, 2], [1, 3, 2], [1, 1, 2], [1, 9, 2]]
import cPickle as pickle
import numpy as np

readed_fpkls = {}

def combine_score(tags, fpkls, score_weights):
	source_num = len(tags)
	exp_num = len(score_weights)
	assert source_num == len(fpkls)
	print("combine_score testing {}, {}, {}".format(tags, fpkls, score_weights))

	pkls = []
	for fpkl in fpkls:
		print("reading {}".format(fpkl))
		readed_pkl = None
		if fpkl in readed_fpkls:
			print("readed")
			readed_pkl = readed_fpkls[fpkl]
		else:
			with open(fpkl, "rb") as f:
				readed_pkl = pickle.load(f)
				readed_fpkls[fpkl] = readed_pkl
		pkls.append(readed_pkl)
	print("pkl files are all readed")
		
	def softmax(m):
	    #print len(m.shape), m.shape
	    # print "softmax", m.shape
	    assert len(m.shape) == 2 and m.shape[1] == 101
	    copy_m = np.copy(m)
	    for i in range(copy_m.shape[0]):
	        copy_m[i,:] = np.exp(copy_m[i,:])
	        copy_m[i,:] = copy_m[i,:]/sum(copy_m[i,:])
	    return copy_m

	# ATTENTION softmax
	# np.set_printoptions(precision=3)
	# np.set_printoptions(threshold=np.nan)
	# np.set_printoptions(suppress=True)

	for exp_id in range(exp_num):
		total = 0
		correct = 0
		corrects = [0] * source_num

		# kind_total = [0] * 101
		# kind_correct = [0] * 101
		# kind_corrects = {tag: [0] * 101 for tag in tags}

		for i in range(len(pkls[0])):
			label = pkls[0][i][2]
			for source_id in range(1, source_num):
				assert pkls[source_id][i][2] == label
			
			for source_id in range(source_num):
				mean_score = np.mean(softmax(pkls[source_id][i][1]), axis=0)
				pd_label = np.argmax(mean_score)
				if pd_label == label:
					# kind_corrects[tags[j]][label] += 1
					corrects[source_id] += 1

			sum_score = np.sum([pkls[source_id][i][1] * score_weights[exp_id][source_id] for source_id in range(source_num)], axis=0)
			sum_score /= np.sum(score_weights[exp_id])
			mean_score = np.mean(softmax(sum_score), axis=0)
			pd_label = np.argmax(mean_score)
			if pd_label == label:
				# kind_correct[label] += 1
				correct += 1
			
			total +=1
			# kind_total[label] += 1

		print("exp {}: {},{}".format(exp_id, tags, score_weights[exp_id]))
		print("weighted sum: {}/{} {}".format(correct, total, correct/float(total)))
		for source_id in range(source_num):
			print("{}: {}/{} {}".format(tags[source_id], corrects[source_id], total, corrects[source_id]/float(total)))
		print("\n")
