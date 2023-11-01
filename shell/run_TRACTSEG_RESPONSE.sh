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
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-TCRc007_ses-1"
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


for val in ${StringArray[@]}; do

        BASEPATH=$val
        echo basepath provided is: $BASEPATH

        ## extract brain mask
        datapath=$BASEPATH/data.nii.gz
        bvalpath=$BASEPATH/bvals
        bvecpath=$BASEPATH/bvecs
        PROCESSPATH=$BASEPATH
        OUTPUTPATH=$BASEPATH/tractseg_outputs
        mkdir -p $OUTPUTPATH

        TractSeg -i $datapath -o $OUTPUTPATH --bvals $bvalpath --bvecs $bvecpath --raw_diffusion_input



done
echo DONE WITH EVERYTHING...
