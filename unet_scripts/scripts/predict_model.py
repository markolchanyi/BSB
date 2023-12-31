import numpy as np 
import sys

sys.path.append('/autofs/space/nicc_003/users/olchanyi/BSB/unet_scripts/unet')

from predict import predict


# keep in list form since you can iterate over multiple sunjects explicitely...implicit will be added soon
#subject_list = ['subject_101107','subject_103111','subject_103414','subject_103818','subject_105115','subject_111716','subject_114419','subject_115320','subject_116524','subject_122620','subject_123925','subject_127630','subject_EXC007','subject_EXC012','subject_EXC030']
#subject_list = ["subject_101107","subject_105115","subject_110411","subject_113619","subject_124422","subject_131722","subject_133928","subject_136833","subject_144832","subject_103414","subject_111312","subject_116524","subject_127933","subject_133019","subject_135225","subject_140925"]
#subject_list = ["subject_101107_shell1000","subject_105115_shell1000","subject_110411_shell1000","subject_113619_shell1000","subject_124422_shell1000","subject_131722_shell1000","subject_133928_shell1000","subject_136833_shell1000","subject_144832_shell1000","subject_103414_shell1000","subject_111312_shell1000","subject_116524_shell1000","subject_133019_shell1000","subject_135225_shell1000","subject_140925_shell1000"]
subject_list = ["subject_C1","subject_C2","subject_C4","subject_EXC007","subject_EXC012","subject_EXC020","subject_EXC021"]
#subject_list = ["subject_110411"]
fs_subject_dir = '/autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/testing_dataset_EXC/validate'
# for now...must be
dataset = 'template'
path_label_list = '../../../data/CRSEG_unet_training_data/SHELLED_9ROI_training_dataset/brainstem_wm_label_list.npy'
model_file = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_ablation_NORGBSTREAMLINES_v1/dice_435.h5'
# model file resolution
resolution_model_file=1.0
# generator mode for prediction data (make sure same as training!)
generator_mode='rgb'
# U-net: number of features in base level (make sure same as training!)
unet_feat_count=24
# U-net: number of levels (make sure same as training!)
n_levels = 5
# U-net: convolution kernel size (make sure same as training!)
conv_size = 3
# U-net: number of features per level multiplier (make sure same as training!)
feat_multiplier = 2
# U-net: number of convolutions per level (make sure same as training!)
nb_conv_per_level = 2
# U-net: activation function (make sure same as training!)
activation='elu'
# (isotropic) dimensions of bounding box to take around thalamus
bounding_box_width = 64
# reference affine
aff_ref = np.eye(4)


predict(subject_list,
            fs_subject_dir,
            dataset,
            path_label_list,
            model_file,
            resolution_model_file,
            generator_mode,
            unet_feat_count,
            n_levels,
            conv_size,
            feat_multiplier,
            nb_conv_per_level,
            activation,
            bounding_box_width,
            aff_ref)
