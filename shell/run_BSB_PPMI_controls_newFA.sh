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

ER_COUNTER=0
CASE_COUNTER=0

SEARCH_DIR=/autofs/space/nicc_003/users/olchanyi/data/PPMI/nifti/ppmi_controls


# Create an empty array
StringArray=()
noise_txt_file="./PPMI_controls_noise.txt"

# Loop through each item in the directory
for item in "$SEARCH_DIR"/*; do
    # Check if the item is a directory
    if [ -d "$item" ]; then
        # Add the directory to the array
        StringArray+=("$item")
    fi
done



for val in ${StringArray[@]}; do
        # Find the subdirectory that starts with "2"
        for SUBDIR in $(find "$val/DTI_gated" -maxdepth 1 -mindepth 1 -type d); do
            # Get just the name of the subdirectory, without its path
            current_name=$(basename "$SUBDIR")

            # Check if the subdirectory name starts with "2"
            if [[ $current_name == 2* ]]; then
            # Set SUBDIR_NAME to the current directory name and break the loop
                SUBDIR_NAME=$current_name
                break
            fi
	done

	BASEPATH=$val/DTI_gated/$SUBDIR_NAME
        echo basepath provided is: $BASEPATH

        ## extract brain mask
        datapath=$BASEPATH/data_preprocessed.nii.gz
        bvalpath=$BASEPATH/bvals
        bvecpath=$BASEPATH/bvecs
        PROCESSPATH=$BASEPATH
        OUTPUTPATH=$BASEPATH/bsb_outputs_attention


        # apply denoising for noise map??
        denoise=False

        if [ "$denoise" = True ]; then
            echo "starting mrtrix denoising"
            dwi2mask $BASEPATH/data_raw.nii.gz $OUTPUTPATH/raw_brain_mask.nii.gz -fslgrad $bvecpath $bvalpath -nthreads 30
            mrconvert $BASEPATH/data_raw.nii.gz $OUTPUTPATH/dwi_raw.mif -fslgrad $bvecpath $bvalpath -nthreads 30
            dwiextract $OUTPUTPATH/dwi_raw.mif - -bzero | mrmath - mean $OUTPUTPATH/lowb_raw.nii.gz -axis 3 -nthreads 30
            dwidenoise $BASEPATH/data_raw.nii.gz $OUTPUTPATH/data_raw_denoised.nii.gz -noise $OUTPUTPATH/raw_noise_map.nii.gz -nthreads 30
            std_dev=$(mrstats $OUTPUTPATH/raw_noise_map.nii.gz -output std -mask $OUTPUTPATH/raw_brain_mask.nii.gz)
            mean_noise=$(mrstats $OUTPUTPATH/raw_noise_map.nii.gz -output mean -mask $OUTPUTPATH/raw_brain_mask.nii.gz)
            mean_signal=$(mrstats $OUTPUTPATH/lowb_raw.nii.gz -output mean -mask $OUTPUTPATH/raw_brain_mask.nii.gz)

            snr=$(echo "scale=2; $mean_signal / $mean_noise" | bc)

            echo SNR is $snr    mean is $mean_signal     std is $mean_noise
            echo $snr >> $noise_txt_file

            rm $OUTPUTPATH/dwi_raw.mif
        else
            echo "not applying denoising"
        fi


        ## check for existance of raw dwi path
        if [ -e $datapath ]
        then
            	echo data path $datapath found
        else
            	echo no data file found
                let ER_COUNTER=ER_COUNTER+1
        fi
        # ----------- mrtrix BSB preprocessing script ----------- #
        if [ -e $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm.nii.gz ]
        then
            	echo "Trackgen outputs already exist...skipping"
        else
            	python ../scripts/trackgen_dev.py \
                        --datapath $datapath \
                        --bvalpath $bvalpath \
                        --bvecpath $bvecpath \
                        --cropsize 64 \
                        --output $OUTPUTPATH \
                        --num_threads 30 \
                        --use_fine_labels False
        fi


        # apply new FA correction?
        FA_correct=True
        if [ "$FA_correct" = True ]; then
            echo "applying FA normalization correction"
            python ../scripts/correct_fa.py --output_path $OUTPUTPATH
        fi


        # ----------- Unet WM segmentation script ----------- #
        if [ -e $OUTPUTPATH/unet_predictions_newfa/unet_results/wmunet.crfseg.mgz ]
        then
            	echo "Unet segmentation outputs already exist...skipping"
        else
            	python ../scripts/unet_wm_predict.py \
                        --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_attention_v10/dice_480.h5 \
                        --output_path $OUTPUTPATH/unet_predictions_newfa \
                        --lowb_file $OUTPUTPATH/lowb_1mm_cropped_norm.nii.gz \
                        --fa_file $OUTPUTPATH/fa_1mm_cropped_norm.nii.gz \
                        --tract_file $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm.nii.gz \
                        --label_list_path /autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/SHELLED_9ROI_training_dataset/brainstem_wm_label_list.npy
        fi
        let CASE_COUNTER=CASE_COUNTER+1


        OUTPUTPATH=$BASEPATH/tractseg_outputs
        if [ -e $OUTPUTPATH/bundle_segmentations/CC.nii.gz ]
        then
            	echo "Tractseg outputs already exist...skipping"
        else
            	TractSeg -i $datapath -o $OUTPUTPATH --bvals $bvalpath --bvecs $bvecpath --raw_diffusion_input
        fi
        echo -------- NUMBER OF CASES DONE ----------
        echo            $CASE_COUNTER
        echo ----------------------------------------

done
echo DONE WITH EVERYTHING...
echo NUMBER OF ERRORS...$ER_COUNTER
