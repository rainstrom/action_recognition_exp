import sys
sys.path.insert(0, '../')
from new_combine_score import combine_score

# combine_score(tags, fpkls, score_weights)

tags = ["rgb_original", "flow_original"]
fpkls = ["../pkl/full_test_rgb_original_no_mirror_no_crop_seg25.pkl", \
		"../pkl/full_test_flow_original_no_mirror_no_crop_seg25.pkl"]
score_weights = [[1, 2], [1, 1], [2, 1], [1, 3], [3, 1]]
combine_score(tags, fpkls, score_weights)

tags = ['rgb_original', "conv1_1minus_rgb5"]
fpkls = ["../pkl/full_test_rgb_original_no_mirror_no_crop_seg25.pkl", \
	"../pkl/full_test_conv1_1minus_rgb5_initfromvgg_iter30000_mirror_crop_seg25.pkl"]
score_weights = [[1, 2], [1, 1], [2, 1], [1, 3], [3, 1]]
combine_score(tags, fpkls, score_weights)

tags = ["rgb_original_mc", "flow_original_mc"]
fpkls = ["../pkl/full_test_rgb_fc8_mirror_crop_seg25.pkl", \
	"../pkl/full_test_flow_fc8_mirror_crop_seg25.pkl"]
score_weights = [[1, 2], [1, 1], [2, 1], [1, 3], [3, 1]]
combine_score(tags, fpkls, score_weights)
