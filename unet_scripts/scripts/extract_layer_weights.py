import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from scipy import ndimage
sys.path.append('../../unet_scripts')
import unet.utils as utils
import unet.models as models
from unet.dcrf_gradient import dense_crf_inference
import nibabel as nib

unet_input_shape = [64, 64, 64, 5]
unet_feat_count=24
n_levels=5
conv_size=3
n_labels=17
feat_multiplier=2
nb_conv_per_level=2
activation='elu'

unet_model = models.unet(nb_features=unet_feat_count,
                             input_shape=unet_input_shape,
                             nb_levels=n_levels,
                             conv_size=conv_size,
                             nb_labels=n_labels,
                             feat_mult=feat_multiplier,
                             nb_conv_per_level=nb_conv_per_level,
                             conv_dropout=0,
                             batch_norm=-1,
                             activation=activation,
                             attention_gating=True,
                             input_model=None)

unet_model.load_weights("/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/model_shelled_attention_v2/dice_510.h5", by_name=True)

print(unet_model.summary())

attention_layer_names = [layer.name for layer in unet_model.layers if 'merge_5' in layer.name]

for layer in unet_model.layers:
    print(layer.name)

print("attention layer name: ", attention_layer_names)

# Initialize an empty dictionary to hold the weights
attention_gate_weights = {}

# Loop through each attention layer and extract its weights

for layer in unet_model.layers:
    if 'merge_6' in layer.name:
        # Assuming you want the weights (not biases) which is usually the first element
        weights = layer.get_weights()
        print(weights)
        np.save('attention_weights.npy', weights)
        print(f'Saved weights of layer {layer.name} to {layer.name}_weights.npy')
        affine = np.eye(4)

        # Create the NIfTI image using the array and the affine matrix
        nifti_image = nib.Nifti1Image(weights, affine)

        # Save the NIfTI image to a file
        nib.save(nifti_image, 'attention_weights.nii.gz')


print("done!")
