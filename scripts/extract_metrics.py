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




"""
MrTrix-based probabalistic tractography preprocessing pipeline for brainstem WM bundles

Usage:


Author:
Mark D. Olchanyi -- 03.17.2023
"""


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

try:
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

    if scrape:
        print("basic scraping for raw DWI file, as well as bval and bvec files...")
        for file in os.listdir(r'F:'):
            if file == "dwi.nii.gz" or file == "data.nii.gz":
                   print(os.path.join(r'F:', file))
    else:
        if not os.path.exists(os.path.join(scratch_dir,"data.nii.gz")):
            os.system("cp -v " + args.datapath + " " + os.path.join(scratch_dir,"data.nii.gz"))
        if not os.path.exists(os.path.join(scratch_dir,"dwi.bval")):
            os.system("rsync -av " + bval_path + " " + os.path.join(scratch_dir,"dwi.bval"))
        if not os.path.exists(os.path.join(scratch_dir,"dwi.bvec")):
            os.system("rsync -av " + bvec_path + " " + os.path.join(scratch_dir,"dwi.bvec"))



    ##### Initial MRTRIX calls #####

    ## perform wrapped FSL preprocessing, requires a DICOM header json to get phase-encoding direction
    if fsl_preprocess:
        #assert dcm_json_header_path != None, 'No DICOM header json file provided. This is required for FSL preprocessing!'
        if dcm_json_header_path != None:
            print("---------- STARTING FSL PREPROCESSING (Eddy + Motion Correction) -----------")
            os.system("dwifslpreproc " + os.path.join(scratch_dir,"data.nii.gz") + " " + os.path.join(scratch_dir,"dwi.mif") + " -json_import " + dcm_json_header_path + " -rpe_header -fslgrad " + os.path.join(scratch_dir,"dwi.bvec") + " " + os.path.join(scratch_dir,"dwi.bval"))
        else:
            print("no header provided, performing nieve preprocessing (not recommended)...")
        print("Finished FSL preprocessing!")

    ## convert dwi to MIF format
    if not os.path.exists(os.path.join(scratch_dir,"dwi.mif")) and not fsl_preprocess:
        os.system("mrconvert " + os.path.join(scratch_dir,"data.nii.gz") + " " + os.path.join(scratch_dir,"dwi.mif") + " -fslgrad " + os.path.join(scratch_dir,"dwi.bvec") + " " + os.path.join(scratch_dir,"dwi.bval") + " -nthreads " + str(num_threads) + " -force")


    # extract header voxel resolution and match it to HCP data (1.25mm iso) and
    # find out if single-shell or not to degermine which FOD algorithm to use.
    os.system("mrinfo -json_all " + os.path.join(scratch_dir,"header.json") + " " + os.path.join(scratch_dir,"dwi.mif") + " -force")
    vox_resolution = get_header_resolution(os.path.join(scratch_dir,"header.json"))
    print("header resolution is " + str(vox_resolution) + " mm")
    shell_count = count_shells(os.path.join(scratch_dir,"header.json"))
    single_shell = shell_count <= 2
    print("...single_shell mode is " + str(single_shell))

    if (vox_resolution > 1.05) or (vox_resolution < 0.95):
        print_no_newline("Resolution is out of bounds!! Regridding dwi to 1mm iso...")
        os.system("mrgrid " + os.path.join(scratch_dir,"dwi.mif") + " regrid -vox 1.0 " + os.path.join(scratch_dir,"dwi_regridded_1mm.mif") + " -force")
        os.system("rm " + os.path.join(scratch_dir,"dwi.mif"))
        os.system("mv " + os.path.join(scratch_dir,"dwi_regridded_1mm.mif") + " " + os.path.join(scratch_dir,"dwi.mif"))
        print("done")

    ## extract mean b0 volume
    print_no_newline("extracting temporary b0...")
    if not os.path.exists(os.path.join(scratch_dir,"mean_b0.mif")):
        os.system("dwiextract " + os.path.join(scratch_dir,"dwi.mif") + " - -bzero | mrmath - mean " + os.path.join(scratch_dir,"mean_b0.mif") + " -axis 3 -force")
        os.system("mrconvert " + os.path.join(scratch_dir,"mean_b0.mif") + " " + os.path.join(output_dir,"lowb_1mm.nii.gz")) # move all relevent volumes to output dir
        os.system("mrconvert " + os.path.join(scratch_dir,"mean_b0.mif") + " " + os.path.join(scratch_dir,"mean_b0.nii.gz") + " -datatype float32 -force")
        ## calculate all scalar volumes from tensor fit and move to output
        os.system("dwi2tensor " + os.path.join(scratch_dir,"dwi.mif") + " " + os.path.join(scratch_dir,"dwi_dt.mif") + " -nthreads " + str(num_threads) + " -force")
        os.system("tensor2metric " + os.path.join(scratch_dir,"dwi_dt.mif") + " -adc " + os.path.join(output_dir,"md_1mm.nii.gz") + " -force")
        #os.system("tensor2metric -dkt " + os.path.join(scratch_dir,"dkt.mif") + " -mk " + os.path.join(output_dir,"mk_1mm.nii.gz") + " -rk " + os.path.join(output_dir,"rk_1mm.nii.gz") + " -ak " + os.path.join(output_dir,"ak_1mm.nii.gz") + " -force")
    print("done")



    #### delete scratch directory
    print_no_newline("deleting scratch directory... ")
    #shutil.rmtree(scratch_dir)
    print("done")
    print("finished case mrtrix and fsl preprocessing \n\n")

except:
    traceback.print_exc()
    print("some exception has occured!!!!")
    print_no_newline("deleting scratch directory...")
    #if os.path.exists(scratch_dir):
    #    shutil.rmtree(scratch_dir)
    print("exiting")

