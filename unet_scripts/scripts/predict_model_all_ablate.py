import numpy as np
import sys

sys.path.append('/autofs/space/nicc_003/users/olchanyi/BSB/unet_scripts/unet')

from predict import predict


# keep in list form since you can iterate over multiple sunjects explicitely...implicit will be added soon
#subject_list = ["subject_C1_rotated"]
#subject_list = ["subject_C1_rotated","subject_C2","subject_C4","subject_EXC007","subject_EXC012","subject_EXC020","subject_EXC021"]
subject_list = ["subject_101107","subject_105115","subject_111312","subject_116524","subject_127933","subject_133019","subject_135225","subject_140925","subject_103414","subject_110411", "subject_113619","subject_124422","subject_131722","subject_133928","subject_136833","subject_144832"]
#subject_list = ["subject_002_S_4213","subject_002_S_4225","subject_002_S_6009","subject_002_S_6030","subject_002_S_6053","subject_003_S_6259","subject_003_S_6260","subject_003_S_6307","subject_012_S_4643","subject_016_S_4951"]

fs_subject_dir = '/autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/testing_dataset_HCP100/validate'
# for now...must be
dataset = 'template'
path_label_list = '../../../data/CRSEG_unet_training_data/SHELLED_9ROI_training_dataset/brainstem_wm_label_list.npy'

model_file_orig = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_attention_v10/dice_480.h5'
model_file_attentionablate = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_ablation_NOATTENTION_v1/dice_435.h5'
model_file_rgbablate = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_ablation_NORGBSTREAMLINES_v1/dice_540.h5'
model_file_v1 = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_ablation_COLORFA_v1/dice_600.h5'


predict_orig = True
predict_noattention = True
predict_norgb = True
predict_v1 = True

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


if predict_orig:
    predict(subject_list,
                fs_subject_dir,
                dataset,
                path_label_list,
                model_file_orig,
                resolution_model_file,
                generator_mode,
                unet_feat_count,
                n_levels,
                conv_size,
                feat_multiplier,
                nb_conv_per_level,
                activation,
                bounding_box_width,
                aff_ref,
                attention=True,
                ablate_rgb=False,
                use_v1=False,
                outputpath="results_bsb_rawres")

if predict_noattention:
    predict(subject_list,
                fs_subject_dir,
                dataset,
                path_label_list,
                model_file_attentionablate,
                resolution_model_file,
                generator_mode,
                unet_feat_count,
                n_levels,
                conv_size,
                feat_multiplier,
                nb_conv_per_level,
                activation,
                bounding_box_width,
                aff_ref,
                attention=False,
                ablate_rgb=False,
                use_v1=False,
                outputpath="results_noattention")


if predict_norgb:
    predict(subject_list,
                fs_subject_dir,
                dataset,
                path_label_list,
                model_file_rgbablate,
                resolution_model_file,
                generator_mode,
                unet_feat_count,
                n_levels,
                conv_size,
                feat_multiplier,
                nb_conv_per_level,
                activation,
                bounding_box_width,
                aff_ref,
                attention=True,
                ablate_rgb=True,
                use_v1=False,
                outputpath="results_norgb")


if predict_v1:
    predict(subject_list,
                fs_subject_dir,
                dataset,
                path_label_list,
                model_file_v1,
                resolution_model_file,
                generator_mode,
                unet_feat_count,
                n_levels,
                conv_size,
                feat_multiplier,
                nb_conv_per_level,
                activation,
                bounding_box_width,
                aff_ref,
                attention=True,
                ablate_rgb=False,
                use_v1=True,
                outputpath="results_v1")

