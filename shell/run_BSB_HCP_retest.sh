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
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/103818"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/111312"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/115320"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/125525"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/135528"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/139839"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/144226"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/149337"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/151526"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/169343"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/175439"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/185442"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/192439"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/195041"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/200614"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/250427"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/341834"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/562345"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/601127"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/660951"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/783462"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/861456"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/917255"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/105923"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/114823"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/122317"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/130518"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/137128"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/143325"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/146129"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/149741"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/158035"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/172332"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/177746"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/187547"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/194140"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/200109"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/204521"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/287248"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/433839"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/599671"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/627549"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/662551"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/859671"
			"/autofs/space/nicc_003/users/olchanyi/data/HCP_retest/877168")


for val in ${StringArray[@]}; do
        BASEPATH=$val
        echo basepath provided is: $BASEPATH

        ## extract brain mask
        datapath=$BASEPATH/T1w/Diffusion/data.nii.gz
        bvalpath=$BASEPATH/T1w/Diffusion/bvals
        bvecpath=$BASEPATH/T1w/Diffusion/bvecs
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

