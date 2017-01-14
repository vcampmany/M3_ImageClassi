
def extract_pyramid_bins(levels, kpt, des, dimensions):
	'''
		dimensions: [min_x,min_y,max_x,max_y] indicates the lower an upper bounds of the level
	'''

	keypoints, descriptors = [], []

	if levels == []: # end recursivity
		return keypoints, descriptors
	else:
		x_divisions, y_divisions = levels[0]

		min_limit_x, min_limit_y = dimensions[0:2]
		max_limit_x, max_limit_y = dimensions[2:4]

		x_step = (max_limit_x-min_limit_x) / float(x_divisions)
		y_step = (max_limit_y-min_limit_y) / float(y_divisions)

		for x_div in range(x_divisions):
			for y_div in range(y_divisions):
				# current bin kps and des
				bin_kpt = []
				bin_des = []

				min_x,max_x = min_limit_x + x_step*x_div, min_limit_x + x_step*(x_div+1)
				min_y,max_y = min_limit_y + y_step*y_div, min_limit_y + y_step*(y_div+1)

				# get current bin keypoints and descriptors
				for i, kp in enumerate(kpt):
					# check if this keypoint belongs to the current bin
					if (kp.pt[0] >= min_x and kp.pt[0] < max_x) and (kp.pt[1] >= min_y and kp.pt[1] < max_y):
						# it belongs!
						bin_kpt.append(kpt[i])
						bin_des.append(des[i])

				# append current bin keypoints and descriptors
				keypoints.append(bin_kpt)
				descriptors.append(bin_des)

				# get keypoints and descriptors corresponding to deeper levels
				level_dimensions = [min_x, min_y, max_x, max_y]
				lower_kps, lower_des = extract_pyramid_bins(levels[1:], bin_kpt, bin_des, level_dimensions)

				keypoints += lower_kps
				descriptors += lower_des

	return keypoints, descriptors