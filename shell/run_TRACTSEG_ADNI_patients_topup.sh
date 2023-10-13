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
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6264"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/003_S_6833"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/016_S_6708"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/033_S_6976"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/036_S_4430"
                        "/autofs/space/nicc_003/users/olchanyi/data/ADNI3_2mm/subject_AD_nii/168_S_6142")


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
        OUTPUTPATH=$BASEPATH/tractseg_outputs
        mkdir -p $OUTPUTPATH


        TractSeg -i $datapath -o $OUTPUTPATH --bvals $bvalpath --bvecs $bvecpath --raw_diffusion_input



done
echo DONE WITH EVERYTHING...
