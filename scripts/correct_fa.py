from utils import rescale_intensities, parse_args_fa
import nibabel as nib
import numpy as np
import os

args = parse_args_fa()
output_dir = args.output_path


vol_fa = nib.load(os.path.join(output_dir,"fa_1mm_cropped.nii.gz"))
vol_fa_np = vol_fa.get_fdata()

vol_fa_np_normalized = rescale_intensities(vol_fa_np,factor=5)

output_img_fa = nib.Nifti1Image(vol_fa_np_normalized, vol_fa.affine, vol_fa.header)
nib.save(output_img_fa, os.path.join(output_dir,"fa_1mm_cropped_norm.nii.gz"))
print("done with correction")
