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
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_1261"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_1280"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_4213"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_4225"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_6007"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_6009"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_6030"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_6053"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_6066"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_6103"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/002_S_6404"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_4288"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_4350"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_4441"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_4644"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_4872"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_4900"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_6014"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_6067"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_6256"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_6259"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_6260"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/003_S_6307"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/011_S_0021"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/011_S_4105"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/011_S_4278"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/011_S_6367"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/011_S_6418"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/011_S_6465"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/011_S_7028"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/012_S_4643"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_4951"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_4952"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6381"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6790"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6834"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6853"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6892"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6931"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6934"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6943"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/016_S_6971"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/020_S_6185"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/020_S_6282"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/020_S_6449"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/020_S_6470"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/020_S_6504"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/020_S_6513"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/020_S_6566"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/024_S_4084"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/024_S_6005"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/024_S_6184"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_0677"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_1169"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_4277"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_4429"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_6211"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_6294"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_6699"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_6701"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_6709"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/032_S_6717"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_0734"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_1098"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_6266"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_6298"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_6352"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_6572"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_6969"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_7100"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/033_S_7114"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/035_S_4464"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/035_S_6156"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/036_S_6088"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/037_S_4028"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/037_S_4308"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/037_S_4410"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/037_S_6032"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/037_S_6046"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/037_S_6115"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/037_S_6144"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_4037"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_4200"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_4427"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6159"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6192"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6226"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6292"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6314"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6354"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6401"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6447"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/041_S_6785"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/068_S_0127"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/068_S_0210"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/068_S_0473"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_control_nii/068_S_4340")


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

        ## check for existance of raw dwi path
        if [ -e $datapath ]
        then
            	echo data path $datapath found
        else
            	echo no data file found
                let ER_COUNTER=ER_COUNTER+1
        fi
        # ----------- mrtrix BSB preprocessing script ----------- #
        if [ -e $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm_new_new.nii.gz ]
        then
            	echo "Trackgen outputs already exist...skipping"
        else
            	python ../scripts/extract_metrics.py \
                        --datapath $datapath \
                        --bvalpath $bvalpath \
                        --bvecpath $bvecpath \
                        --cropsize 64 \
                        --output $OUTPUTPATH \
                        --num_threads 50 \
                        --use_fine_labels False
        fi




        # ----------- Unet WM segmentation script ----------- #
        #if [ -e $OUTPUTPATH/unet_predictions/unet_results/wmunet.crfseg.mgz ]
        #then
        #    	echo "Unet segmentation outputs already exist...skipping"
        #else
        #    	python ../scripts/unet_wm_predict.py \
        #                --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_attention_v1/dice_510.h5 \
        #                --output_path $OUTPUTPATH/unet_predictions \
        #                --lowb_file $OUTPUTPATH/lowb_1mm_cropped_norm.nii.gz \
        #                --fa_file $OUTPUTPATH/fa_1mm_cropped_norm.nii.gz \
        #                --tract_file $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm.nii.gz \
        #                --label_list_path /autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/SHELLED_9ROI_training_dataset/brainstem_wm_label_list.npy
        #fi
        let CASE_COUNTER=CASE_COUNTER+1

        echo -------- NUMBER OF CASES DONE ----------
        echo            $CASE_COUNTER
        echo ----------------------------------------


done
echo DONE WITH EVERYTHING...
echo NUMBER OF ERRORS...$ER_COUNTER
