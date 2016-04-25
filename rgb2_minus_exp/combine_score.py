from ../combine_score import combine_score

fpkl1 = "./full_test_rgb_original_no_mirror_no_crop_seg25.pkl"
fpkl2 = "./conv1_1minus_rgb5/full_test_conv1_1minus_rgb5_initfromvgg_iter30000_no_mirror_no_crop_seg25.pkl"

# combine_score(tags, fpkls, score_weights):
combine_score(["rgb_original", "conv1_1minus_rgb5"], [fpkl1, fpkl2])
