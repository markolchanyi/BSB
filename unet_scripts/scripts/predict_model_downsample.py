import numpy as np
import sys

sys.path.append('/autofs/space/nicc_003/users/olchanyi/BSB/unet_scripts/unet')

from predict import predict


# keep in list form since you can iterate over multiple sunjects explicitely...implicit will be added soon

#subject_list = ["subject-TCRp017-acute","subject-TCRp017-chronic"]
#subject_list = ["subject_C1","subject_C2","subject_C4","subject_EXC007","subject_EXC012","subject_EXC020","subject_EXC021"]
#subject_list = ["subject_101107","subject_105115","subject_111312","subject_116524","subject_127933","subject_133019","subject_135225","subject_140925","subject_103414","subject_110411","subject_113619","subject_124422","subject_131722","subject_133928","subject_136833","subject_144832"]
subject_list = ["subject_002_S_4213","subject_002_S_4225","subject_002_S_6009","subject_002_S_6030","subject_002_S_6053","subject_003_S_6259","subject_003_S_6260","subject_003_S_6307","subject_012_S_4643","subject_016_S_4951"]

fs_subject_dir = '/autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/testing_dataset_ADNI/validate'
# for now...must be
dataset = 'template'
path_label_list = '../../../data/CRSEG_unet_training_data/SHELLED_9ROI_training_dataset/brainstem_wm_label_list.npy'

model_file_orig = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_attention_v10/dice_465.h5'
#model_file_attentionablate = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_ablation_NOATTENTION_v1/dice_435.h5'
#model_file_rgbablate = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_ablation_NORGBSTREAMLINES_v1/dice_435.h5'
#model_file_v1 = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_ablation_COLORFA_v1/dice_435.h5'


#predict_orig = True
#predict_noattention = True
#predict_norgb = True
#predict_v1 = True

# model file resolution
resolution_model_file=1.0
generator_mode='rgb'
unet_feat_count=24
n_levels = 5
conv_size = 3
feat_multiplier = 2
nb_conv_per_level = 2
activation='elu'
bounding_box_width = 64
aff_ref = np.eye(4)

ds_factor_list = [1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4]
ds_factor_list_names = ["ds1","ds2","ds3","ds4","ds5","ds6","ds7","ds8","ds9","ds10","ds11","ds12","ds13"]


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
                outputpath="results_bsb_rawres",
                ds_factor=None)


for count, ds_factor in enumerate(ds_factor_list):
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
                outputpath="results_bsb_" + ds_factor_list_names[count],
                ds_factor=ds_factor)
