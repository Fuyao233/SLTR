Readme file for fusion replication paper - stimulus

156ImageStimuliSet.zip:
The folder 156ImageStimuliSet contains all 156 .jpg images and a .mat file, visual_stimuli156.mat.
visual_stimuli156.mat contains the variable visual_stimuli156, a structure with 3 fields: pixel_values,
twinset, and category. The pixel_values field is pixels x pixels x 3 and contains the RGB values at each
pixel location. The twinset field is either 1 or 2 for each image, corresponding to which twinset the image
belongs to. The category field is a string of describing which of the five categories this image belongs to:
'animals', 'objects', 'scenes', 'people', or 'faces'. The order of the 156 .jpg images and the information in
visual_stimuli156 are the same (i.e. visual_stimuli156(1) corresponds to 001.jpg). Note that every image belongs
to both a category and a twinset.

category_indices.mat:
category_indices.mat contains five variables: animals, faces, objects, people, and scenes. These variables are
vectors corresponding to which images belong to which category. For example, animals is a vector 1-28, meaning
images 001.jpg - 028.jpg are all animals.

twinset_indices.mat:
twinset_indices.mat contains two variables: twinset1_indices and twinset2_indices. These variables are
vectors corresponding to which images belong to which twinset. For example, twinset1_indices contain values
1, 4, 5, 8...etc meaning that images 001.jpg, 004.jpg, 005.jpg, 008.jpg and so forth belong to twinset 1.