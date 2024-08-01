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
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c001"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c002"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c004"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c005"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c006"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c007"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c008"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c010"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c012"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c013"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c014"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c015"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c016"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c017"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c018"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c019"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc007_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc008_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc010_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc011_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc012_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc013_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc014_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc015_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc016_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc017_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc018_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc019_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc020_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc021_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc022_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc023_ses-1"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp001_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp003_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp005_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp006_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp007_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp008_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp009_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp010_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp011_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp012_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp013_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp014_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp015_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp016_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp017_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp018_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp022_ses-early"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRp023_ses-early")

noise_txt_file="./RESPONSE_noise.txt"

for val in ${StringArray[@]}; do
        BASEPATH=$val
        echo basepath provided is: $BASEPATH

        ## extract brain mask
        datapath=$BASEPATH/data.nii.gz
        bvalpath=$BASEPATH/bvals
        bvecpath=$BASEPATH/bvecs
        PROCESSPATH=$BASEPATH
        OUTPUTPATH=$BASEPATH/bsb_outputs_attention




        # apply denoising for noise map??
        denoise=True

        if [ "$denoise" = True ]; then
            echo "starting mrtrix denoising"
            dwi2mask $BASEPATH/data.nii.gz $OUTPUTPATH/raw_brain_mask.nii.gz -fslgrad $bvecpath $bvalpath -nthreads 30 -force
            mrconvert $BASEPATH/data.nii.gz $OUTPUTPATH/dwi_raw.mif -fslgrad $bvecpath $bvalpath -force -nthreads 30
            dwiextract $OUTPUTPATH/dwi_raw.mif - -bzero | mrmath - mean $OUTPUTPATH/lowb_raw.nii.gz -axis 3 -nthreads 30 -force
            dwidenoise $BASEPATH/data.nii.gz $OUTPUTPATH/data_raw_denoised.nii.gz -noise $OUTPUTPATH/raw_noise_map.nii.gz -nthreads 30 -force
            std_dev=$(mrstats $OUTPUTPATH/raw_noise_map.nii.gz -output std -mask $OUTPUTPATH/raw_brain_mask.nii.gz)
            mean_noise=$(mrstats $OUTPUTPATH/raw_noise_map.nii.gz -output mean -mask $OUTPUTPATH/raw_brain_mask.nii.gz)
            mean_signal=$(mrstats $OUTPUTPATH/lowb_raw.nii.gz -output mean -mask $OUTPUTPATH/raw_brain_mask.nii.gz)

            snr=$(echo "scale=2; $mean_signal / $mean_noise" | bc)

            echo SNR is $snr    mean signal is $mean_signal     mean noise is $mean_noise
            echo $snr >> $noise_txt_file
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
                        --num_threads 50 \
                        --use_fine_labels False
        fi




        # ----------- Unet WM segmentation script ----------- #
        if [ -e $OUTPUTPATH/unet_predictions_raw/unet_results/wmunet.crfseg.mgz ]
        then
                echo "Unet segmentation outputs already exist...skipping"
        else
                python ../scripts/unet_wm_predict.py \
                        --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_attention_v10/dice_480.h5 \
                        --output_path $OUTPUTPATH/unet_predictions_raw \
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

