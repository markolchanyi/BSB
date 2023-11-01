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

# Declare an array of string with type
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6264"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6833"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6708"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_6976"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/036_S_4430"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6142"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/002_S_1155"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/002_S_4229"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/002_S_4654"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_0908"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_4354"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6258"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6268"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6432"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6606"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6678"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/011_S_4827"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/011_S_6303"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/011_S_6618"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/012_S_4094"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/012_S_6073"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/012_S_6503"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/012_S_6760"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_4902"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_5057"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6771"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6789"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6809"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6816"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6836"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6839"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6904"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6926"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6939"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6949"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_7014"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/032_S_6055"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/032_S_6600"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/032_S_6700"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/032_S_6728"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/032_S_6804"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/032_S_6855"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_6497"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_6705"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_6824"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_6889"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_7066"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_7088"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/035_S_4414"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/036_S_4538"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/036_S_6099"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/037_S_4030"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/037_S_4214"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/037_S_4302"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/037_S_6083"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/037_S_6125"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/041_S_0679"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/041_S_4510"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/041_S_6731"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/068_S_4061"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/094_S_6736"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_4631"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6557"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6576"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6654"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6659"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6685"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6693"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6812"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6880"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/137_S_6919"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/153_S_6679"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6426"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6467"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6541"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6591"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6619"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6634"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6735"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6754"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6815"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6817"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6821"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6827"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6828"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6843"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6851"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6860"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6873"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6874"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6875"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6902"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6921"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6938"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/941_S_4187"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/941_S_6017"
			"/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/941_S_6068")


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
        #        echo "Unet segmentation outputs already exist...skipping"
        #else
        #        python ../scripts/unet_wm_predict.py \
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
