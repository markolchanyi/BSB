#source freeurfer for samseg
#7.3.0 is the most stable version that does not throw a BLAS error
#and can also run synthSR and brainstem_subfield seg
export FREESURFER_HOME=/usr/local/freesurfer/7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh
source /usr/pubsw/packages/mrtrix/env.sh
mrtrixdir=/usr/pubsw/packages/mrtrix/3.0.2/bin
fsldir=/usr/pubsw/packages/fsl/6.0.3/bin
FSLDIR=/usr/pubsw/packages/fsl/6.0.3
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/9.1/lib64
#export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/11.6/lib64:/usr/pubsw/packages/CUDA/11.6/extras/CUPTI/lib64:/usr/pubsw/packages/CUDA/9.0/lib64:/usr/pubsw/packages/CUDA/9.1/lib64

#set -e # The script will terminate after the first line that fails


# Declare an array of string with type
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_7079"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6180"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/037_S_6141"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6903"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6954"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6883"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/036_S_6134"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6479"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6800"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6908"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_7039"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_6697"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/941_S_6052"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_1122"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/037_S_0150"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_7002")



for val in ${StringArray[@]}; do
        for SUBDIR in $(find "$val/Axial_DTI" -maxdepth 1 -mindepth 1 -type d); do
        # Get just the name of the subdirectory, without its path
            SUBDIR_NAME=$(basename "$SUBDIR")
        # Now you can do whatever you want with the subdirectory name
           #echo "$SUBDIR_NAME"
        done

        BASEPATH=$val/Axial_DTI/$SUBDIR_NAME
        echo basepath provided is: $BASEPATH

        ## extract brain mask
        datapath=$BASEPATH/data_preprocessed.nii.gz
        bvalpath=$BASEPATH/bvals
        bvecpath=$BASEPATH/bvecs
        PROCESSPATH=$BASEPATH
        OUTPUTPATH=$BASEPATH/bsb_outputs_attention



        # ----------- mrtrix BSB preprocessing script ----------- #
        if [ -e $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm_new.nii.gz ]
        then
                echo "Trackgen outputs already exist...skipping"
        else
                python ../CRSEG/trackgen_dev.py \
                        --datapath $datapath \
                        --bvalpath $bvalpath \
                        --bvecpath $bvecpath \
                        --cropsize 64 \
                        --output $OUTPUTPATH \
                        --num_threads 50 \
                        --use_fine_labels False
        fi




        # ----------- Unet WM segmentation script ----------- #
        if [ -e $BASEPATH/crseg_outputs_new_medulla/unet_predictions/unet_results/wmunet.seg.mgz ]
        then
                echo "Unet segmentation outputs already exist...skipping"
        else
                python ../CRSEG/unet_wm_predict.py \
                        --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/9ROI_wmb_model_shelled_v10/dice_405.h5 \
                        --output_path $OUTPUTPATH/unet_predictions \
                        --lowb_file $OUTPUTPATH/lowb_1mm_cropped_norm.nii.gz \
                        --fa_file $OUTPUTPATH/fa_1mm_cropped_norm.nii.gz \
                        --tract_file $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm.nii.gz \
                        --label_list_path /autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/SHELLED_9ROI_training_dataset/brainstem_wm_label_list.npy
        fi


done
echo DONE WITH EVERYTHING...
