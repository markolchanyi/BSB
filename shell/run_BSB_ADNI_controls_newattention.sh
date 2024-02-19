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

SEARCH_DIR=/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii


# Create an empty array
StringArray=()

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
        for SUBDIR in $(find "$val/Axial_DTI" -maxdepth 1 -mindepth 1 -type d); do
            # Get just the name of the subdirectory, without its path
            current_name=$(basename "$SUBDIR")

            # Check if the subdirectory name starts with "2"
            if [[ $current_name == 2* ]]; then
            # Set SUBDIR_NAME to the current directory name and break the loop
                SUBDIR_NAME=$current_name
                break
            fi
	done

	BASEPATH=$val/Axial_DTI/$SUBDIR_NAME
        echo basepath provided is: $BASEPATH

        ## extract brain mask
        datapath=$BASEPATH/data_preprocessed.nii.gz
        bvalpath=$BASEPATH/bvals
        bvecpath=$BASEPATH/bvecs
        PROCESSPATH=$BASEPATH
        OUTPUTPATH=$BASEPATH/bsb_outputs_attention

        ## check for existance of raw dwi path
        if [ -e $datapath ]
        then
            	echo data path $datapath found
        else
            	echo no data file found
                let ER_COUNTER=ER_COUNTER+1
        fi
        # ----------- mrtrix BSB preprocessing script ----------- #
        if [ -e $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm_new.nii.gz ]
        then
            	echo "Trackgen outputs already exist...skipping"
        else
            	python ../scripts/trackgen_dev.py \
                        --datapath $datapath \
                        --bvalpath $bvalpath \
                        --bvecpath $bvecpath \
                        --cropsize 64 \
                        --output $OUTPUTPATH \
                        --num_threads 50 \
                        --use_fine_labels False
        fi




        # ----------- Unet WM segmentation script ----------- #
        if [ -e $OUTPUTPATH/unet_predictions_newattention_raw/unet_results/wmunet.crfseg.mgz ]
        then
            	echo "Unet segmentation outputs already exist...skipping"
        else
            	python ../scripts/unet_wm_predict.py \
                        --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_attention_v10/dice_465.h5 \
                        --output_path $OUTPUTPATH/unet_predictions_newattention_raw \
                        --lowb_file $OUTPUTPATH/lowb_1mm_cropped_norm.nii.gz \
                        --fa_file $OUTPUTPATH/fa_1mm_cropped_norm.nii.gz \
                        --tract_file $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm.nii.gz \
                        --label_list_path /autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/SHELLED_9ROI_training_dataset/brainstem_wm_label_list.npy
        fi
        let CASE_COUNTER=CASE_COUNTER+1


        OUTPUTPATH=$BASEPATH/tractseg_outputs
        if [ -e $OUTPUTPATH/bundle_segmentations_TESTTEST/CC.nii.gz ]
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
