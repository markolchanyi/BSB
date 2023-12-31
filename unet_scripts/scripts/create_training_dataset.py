import os,sys,shutil
import numpy as np
import random
import nibabel as nib
from skimage.measure import label
from scipy import ndimage



def rescale_intensities(vol,factor=5):
        vol = vol - vol.mean()
        vol = vol/(factor*vol.std())

        vol += 0.5
        #clip
        vol[vol < 0] = 0
        vol[vol > 1] = 1

        return vol




########
# CURRENTLY ONLY APPLIES FOR HCP!!!
########

data_dir = "/Users/markolchanyi/Desktop/Edlow_Brown/Projects/datasets/HCP/HCP100"
output_dir = "/Users/markolchanyi/Desktop/Edlow_Brown/Projects/datasets/HCP/HCP100/7ROI_training_dataset"
training_dir = os.path.join(output_dir,"train")
validation_dir = os.path.join(output_dir,"validate")
orig_ROI_list = ["ROI00000","ROI00001","ROI00002","ROI00003","ROI00004","ROI00005","ROI00006"]
split_roi = [True,True,True,True,True,True,False] # is this a L/R ROI or not?
right_start_label_num = 1001  # copying thalamic LUT labels for now since FS LUT for these tracts dont exist
left_start_label_num = 2001

## between 0 (all validate) and 1 (all train)
training_ratio = 0.8

if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
os.makedirs(training_dir)
os.makedirs(validation_dir)

## make np label list
label_array = np.zeros((2*len(orig_ROI_list)+1,),dtype=np.int)
label_array[0,] = 0
for ind, roi in enumerate(orig_ROI_list):
        label_array[ind+1,] = right_start_label_num + ind
        label_array[ind+len(orig_ROI_list)+1,] = left_start_label_num + ind

print("label array is: ", label_array)

np.save(os.path.join(output_dir,"brainstem_wm_label_list.npy"),label_array)

data_list = next(os.walk(data_dir))[1]
random.shuffle(data_list)

train_counter = 0
val_counter = 0

for subject_name in data_list:
        subject_dir = os.path.join(data_dir,subject_name)
        if os.path.exists(os.path.join(subject_dir,"crseg_wm_labels/ROI00006.mif")):
                if random.uniform(0, 1) > training_ratio:
                        output_subject_dir = os.path.join(validation_dir,"subject_" + subject_name)
                        val_counter += 1
                else:
                        output_subject_dir = os.path.join(training_dir,"subject_" + subject_name)
                        train_counter += 1

                os.makedirs(output_subject_dir)
                os.makedirs(os.path.join(output_subject_dir,"dmri"))
                os.makedirs(os.path.join(output_subject_dir,"segs"))
                os.makedirs(os.path.join(output_subject_dir,"temp")) # just for temporary conversion stuff

                #seg_file = os.path.join(subject_dir,"training_labels/ROI00.nii.gz")

                for iter, roi in enumerate(orig_ROI_list):
                        roi_mif = roi + ".mif"
                        roi_nifti = roi + ".nii.gz"
                        # convert everything to 64x64x64
                        #os.system("cp " + os.path.join(subject_dir,"crseg_wm_labels",roi_mif) + " ", os.path.join(output_subject_dir,"temp",roi_mif))
                        os.system("mrgrid " + os.path.join(subject_dir,"crseg_wm_labels",roi_mif) + " crop -uniform 3 " + os.path.join(output_subject_dir,"temp",roi_nifti) + " -force")

                        vol = nib.load(os.path.join(output_subject_dir,"temp",roi_nifti))
                        vol_np = vol.get_fdata()

                        if iter == 0:
                                output_vol = np.zeros_like(vol_np)

                        if split_roi[iter]:
                                cc_labels = label(vol_np,connectivity=3)
                                mask_1 = cc_labels == np.argsort(np.bincount(cc_labels.flat, weights=vol_np.flat))[1]
                                mask_2 = cc_labels == np.argsort(np.bincount(cc_labels.flat, weights=vol_np.flat))[2]
                                com1 = ndimage.measurements.center_of_mass(mask_1)
                                com2 = ndimage.measurements.center_of_mass(mask_2)
                                print("mask1 volume: " + str(np.sum(mask_1)) + "        mask2 volume: " + str(np.sum(mask_2)))
                                if abs(np.sum(mask_1)/np.sum(mask_2) - 1) > 0.4:
                                        print("POTENTIALLY BAD L/R PARSING...CHECK VOLUME!!")

                                # from RAS coordinate frame
                                if com1[0] < com2[0]:
                                        mask_right = mask_1
                                        mask_left = mask_2
                                else:
                                        mask_left = mask_1
                                        mask_right = mask_2

                                output_vol[mask_right == 1] = iter + right_start_label_num # make right ROI values 21,22,23...
                                output_vol[mask_left == 1] = iter + left_start_label_num  # make left ROI values 121,122,123...
                        else:
                                cc_labels = label(vol_np,connectivity=3)
                                mask = cc_labels == np.argsort(np.bincount(cc_labels.flat, weights=vol_np.flat))[1]
                                output_vol[mask == 1] = iter + right_start_label_num

                output_img = nib.Nifti1Image(output_vol, vol.affine, vol.header)
                nib.save(output_img, os.path.join(output_subject_dir,"segs","seg.nii.gz"))
                os.system("mri_convert " + os.path.join(output_subject_dir,"segs","seg.nii.gz") + " " + os.path.join(output_subject_dir,"segs","seg.nii.gz") + " -rl " + os.path.join(data_dir,"template","ROI_64_template.nii.gz") + " -rt nearest -odt float")

                b0_file = os.path.join(subject_dir, 'T1w/Diffusion', 'lowb.nii')
                fa_file = os.path.join(subject_dir, 'T1w/Diffusion/dmri', 'dtifit.1K_FA.nii.gz')
                track_file = os.path.join(subject_dir, 'mrtrix_outputs', 'tracts_concatenated_1mm_cropped.mif')

                print("copying volumes for: ", subject_name)
                ## move to respective training folders
                os.system("cp " + b0_file + " " + os.path.join(output_subject_dir,"temp"))
                os.system("cp " + fa_file + " " + os.path.join(output_subject_dir,"temp"))

                os.system("mri_convert " + os.path.join(output_subject_dir,"temp","lowb.nii") + " " + os.path.join(output_subject_dir,"temp","lowb.nii.gz") + " -rl " + os.path.join(data_dir,"template","ROI_64_template.nii.gz") + " -odt float")
                os.system("mri_convert " + os.path.join(output_subject_dir,"temp","dtifit.1K_FA.nii.gz") + " " + os.path.join(output_subject_dir,"temp","FA.nii.gz") + " -rl " + os.path.join(data_dir,"template","ROI_64_template.nii.gz") + " -odt float")

                os.system("mrgrid " + track_file + " crop -uniform 3 " + os.path.join(output_subject_dir,"temp","tracts.nii.gz") + " -force")
                os.system("mri_convert " + os.path.join(output_subject_dir,"temp","tracts.nii.gz") + " " + os.path.join(output_subject_dir,"temp","tracts.nii.gz") + " -rl " + os.path.join(data_dir,"template","ROI_64_template_4D.nii.gz") + " -odt float")


                ####
                #### normalize all volumes!! (0 mean unitary variance)
                ####
                vol_tracts = nib.load(os.path.join(output_subject_dir,"temp","tracts.nii.gz"))
                vol_tracts_np = vol_tracts.get_fdata()

                vol_lowb = nib.load(os.path.join(output_subject_dir,"temp","lowb.nii.gz"))
                vol_lowb_np = vol_lowb.get_fdata()

                vol_fa = nib.load(os.path.join(output_subject_dir,"temp","FA.nii.gz"))
                vol_fa_np = vol_fa.get_fdata()

                ### rescaling
                vol_tracts_np_normalized = rescale_intensities(vol_tracts_np,factor=20)
                vol_lowb_np_normalized = rescale_intensities(vol_lowb_np,factor=5)
                vol_fa_np_normalized = rescale_intensities(vol_fa_np,factor=5)

                print("TRACTS MEAN IS: ", np.mean(vol_tracts_np_normalized))
                print("TRACTS VAR IS: ", np.var(vol_tracts_np_normalized))
                print("LOWB MEAN IS: ", np.mean(vol_lowb_np_normalized))
                print("LOWB VAR IS: ", np.var(vol_lowb_np_normalized))
                print("FA MEAN IS: ", np.mean(vol_fa_np_normalized))
                print("FA VAR IS: ", np.var(vol_fa_np_normalized))

                output_img_tracts = nib.Nifti1Image(vol_tracts_np_normalized, vol_tracts.affine, vol_tracts.header)
                nib.save(output_img_tracts, os.path.join(output_subject_dir,"dmri","tracts.nii.gz"))

                output_img_lowb = nib.Nifti1Image(vol_lowb_np_normalized, vol_lowb.affine, vol_lowb.header)
                nib.save(output_img_lowb, os.path.join(output_subject_dir,"dmri","lowb.nii.gz"))

                output_img_fa = nib.Nifti1Image(vol_fa_np_normalized, vol_tracts.affine, vol_tracts.header)
                nib.save(output_img_fa, os.path.join(output_subject_dir,"dmri","FA.nii.gz"))

                shutil.rmtree(os.path.join(output_subject_dir,"temp"))
                #os.system("mrconvert " + track_file + " " + os.path.join(output_subject_dir,"dmri","tracts_concatenated_1mm_cropped.nii.gz"))
                #os.system("cp " + aseg_file + " " + os.path.join(output_subject_dir,"segs",'subject_' + subject_name + '_1k_DSWbeta.nii.gz'))


print("num training: ", train_counter)
print("nun validation: ", val_counter)
print("FINISHED")
