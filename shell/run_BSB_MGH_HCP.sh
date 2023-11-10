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
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1001"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1002"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1003"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1004"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1005"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1006"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1007"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1008"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1009"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1010"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1011"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1012"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1013"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1014"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1015"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1016"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1017"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1018"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1019"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1020"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1021"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1022"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1023"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1024"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1025"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1026"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1027"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1028"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1029"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1030"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1031"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1032"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1033"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1034"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT/mgh_1025")


for val in ${StringArray[@]}; do
        BASEPATH=$val
        echo basepath provided is: $BASEPATH

        ## extract brain mask
        datapath=$BASEPATH/all_shells/data_corr.nii.gz
        bvalpath=$BASEPATH/all_shells/bvals
        bvecpath=$BASEPATH/all_shells/bvecs
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
                        --num_threads 60 \
                        --use_fine_labels False
        fi




        # ----------- Unet WM segmentation script ----------- #
        if [ -e $OUTPUTPATH/unet_predictions/unet_results/wmunet.crfseg.mgz ]
        then
                echo "Unet segmentation outputs already exist...skipping"
        else
                python ../scripts/unet_wm_predict.py \
                        --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_attention_v1/dice_510.h5 \
                        --output_path $OUTPUTPATH/unet_predictions \
                        --lowb_file $OUTPUTPATH/lowb_1mm_cropped_norm.nii.gz \
                        --fa_file $OUTPUTPATH/fa_1mm_cropped_norm.nii.gz \
                        --tract_file $OUTPUTPATH/tracts_concatenated_1mm_cropped_norm.nii.gz \
                        --label_list_path /autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/SHELLED_9ROI_training_dataset/brainstem_wm_label_list.npy
        fi
        let CASE_COUNTER=CASE_COUNTER+1

        echo -------- NUMBER OF CASES DONE ----------
        echo            $CASE_COUNTER
        echo ----------------------------------------


done
echo DONE WITH EVERYTHING...
echo NUMBER OF ERRORS...$ER_COUNTER

