import os, sys, argparse, datetime, traceback
import shutil
import math
import random
import string
import traceback
import sys
import numpy as np
import multiprocessing as mp
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import is_header_compatible
from utils import print_no_newline, parse_args_mrtrix, count_shells, get_header_resolution, tractography_mask, rescale_intensities


##Parser
args = parse_args_mrtrix()
#------------------- Required Arguments -------------------
local_data_path = args.datapath
bval_path = args.bvalpath
bvec_path = args.bvecpath
crop_size = args.cropsize
output_folder = args.output
dcm_json_header_path = args.json_header_path
fsl_preprocess = args.fsl_preprocess
scrape = args.scrape
unet_segment = args.unet_segment
num_threads = args.num_threads

samseg_path = os.path.join(output_folder,"samseg_labels","")


thal_labels = [10,49]
DC_labels = [28,60]
cort_labels = [18,54]
CB_labels = [7,46]
midbrain_label = 173
pons_label = 174
medulla_label = 175
brainstem_label = 16
hypothal_labels = [801,802,803,804,805,806,807,808,809,810]


if not os.path.exists(os.path.join(output_folder,"brainstem.nii.gz")):
    print("============================================================")
    print("STARTING TRACKEGN \n" + "DWI FILE: " + local_data_path + "\n")
    print("STARTING TRACKEGN \n" + "DWI FILE: " + bval_path + "\n")
    print("STARTING TRACKEGN \n" + "DWI FILE: " + bvec_path)
    print("Crop size: ", str(crop_size))
    print("Using " + str(num_threads) + "threads")
    print("Perform FSL DWI preprocessing?: ", fsl_preprocess)
    print("Perform BSB CNN segmentation?: ", unet_segment)
    print("============================================================")

    letters = string.ascii_lowercase
    scratch_str = "temp_" + ''.join(random.choice(letters) for i in range(10))
    scratch_dir = os.path.join(output_folder,scratch_str,"")
    output_dir = output_folder
    print("All final MRTrix volumes will be dropped in ", output_dir)

    ## --------------- exit clause --------------- ##
    if os.path.exists(os.path.join(output_dir,"tracts_concatenated_1mm_cropped_norm.nii.gz")):
        print("MRTRIX outputs already exists...")

    print("creating temporary scratch directory ", scratch_dir)
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    if not os.path.exists(os.path.join(output_dir,"fs_subj_dir","mri")):
        os.makedirs(os.path.join(output_dir,"fs_subj_dir","mri"))
    os.system("mri_convert " + os.path.join(output_dir,"lowb_synthsr.nii.gz") + " " + os.path.join(output_dir,"fs_subj_dir","mri","T1.mgz"))
    shutil.copy(os.path.join(output_dir,"fs_subj_dir","mri","T1.mgz"), os.path.join(output_dir,"fs_subj_dir","mri","norm.mgz"))
    shutil.copy(os.path.join(samseg_path,"seg.mgz"), os.path.join(output_dir,"fs_subj_dir","mri","aseg.mgz"))
    os.system("mri_convert " + os.path.join(output_dir,"fs_subj_dir","mri","T1.mgz") + " " + os.path.join(output_dir,"fs_subj_dir","mri","T1.mgz") + " -rl " + os.path.join(output_dir,"lowb_1mm.nii.gz") + " -odt float")
    os.system("mri_convert " + os.path.join(output_dir,"fs_subj_dir","mri","norm.mgz") + " " + os.path.join(output_dir,"fs_subj_dir","mri","norm.mgz") + " -rl " + os.path.join(output_dir,"lowb_1mm.nii.gz") + " -odt float")
    if not os.path.exists(os.path.join(output_dir,"brainstem_subfields")):
        os.makedirs(os.path.join(output_dir,"brainstem_subfields"))
    os.system("segment_subregions brainstem --out-dir=" + os.path.join(output_dir,"brainstem_subfields") + " --cross " + os.path.join(output_dir,"fs_subj_dir") + " --threads 80")
    os.system("mri_convert " + os.path.join(output_dir,"brainstem_subfields","brainstemSsLabels.FSvoxelSpace.mgz") + " " + os.path.join(output_dir,"brainstem_subfields","brainstemSsLabels.FSvoxelSpace.mgz") + " -rl " + os.path.join(output_dir,"lowb_1mm.nii.gz")  + " -rt nearest -odt float")

    os.system("mri_binarize --noverbose --i " + os.path.join(output_dir,"brainstem_subfields","brainstemSsLabels.FSvoxelSpace.mgz") + " --o " + os.path.join(output_dir,"midbrain.nii") + " --match " + str(midbrain_label))
    os.system("mri_binarize --noverbose --i " + os.path.join(output_dir,"brainstem_subfields","brainstemSsLabels.FSvoxelSpace.mgz") + " --o " + os.path.join(output_dir,"pons.nii") + " --match " + str(pons_label))
    os.system("mri_binarize --noverbose --i " + os.path.join(output_dir,"brainstem_subfields","brainstemSsLabels.FSvoxelSpace.mgz") + " --o " + os.path.join(output_dir,"medulla.nii") + " --match " + str(medulla_label))

    os.system("mrmath " + os.path.join(output_dir,"midbrain.nii") + " " + os.path.join(output_dir,"pons.nii") + " " + os.path.join(output_dir,"medulla.nii") + " max " + os.path.join(output_dir,"brainstem.nii.gz") + " -force")
else:
    print("case already processed")

