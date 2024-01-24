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
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c016"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c017"
                        "/autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/sub-R2c019")


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
