import os
import tensorflow as tf
import tensorflow_mri as tfmri
from models.variational_network import VarNet, custom_metric_ssim, custom_loss_ssim, custom_loss_mse
from models.layers import FastDVDNet

def default_configs(model_type,DEBUG=False):
    if model_type=='VarNet':
        config_model= { 'rank':2.5,
                'lbd_trainable':True,
                    'num_recon_blocks': 3 if DEBUG else 12, 
                    'kernel_sizes':[3],
                    'unet_layer_sizes': [16,16] if DEBUG else [16,32,64],
                    'block_depth': 2,
                    'activation': lambda x:tf.keras.activations.relu(x,alpha=0.1),
                    'kernel_initializer': tf.keras.initializers.HeUniform(seed=1)}
    elif model_type=='3DUNet':
        config_model= {'filters': [64,92,128],
                    'block_depth': 2,
                    'kernel_size': 3,
                    'activation': lambda x: tf.keras.activations.relu(x,alpha=0.1),
                    'out_channels': 1,
                    'kernel_initializer': tf.keras.initializers.HeUniform(seed=1)}
    elif model_type == 'FastDVDNet':
        config_model={'scales': 3,
                        'block_depth': 2,
                        'base_filters': 32,
                        'kernel_size': 3,
                        'use_deconv': 'PixelShuffle',
                        'rank': 2,
                        'activation': tf.keras.activations.relu,
                        'out_channels': 1,
                        'kernel_initializer': tf.keras.initializers.HeUniform(seed=1),
                        'time_distributed': False,
                        'selected_frame':-1}
    else: 
        config_model=dict()
        print('model_type Unsupported Returning empty config_model')
    return config_model

def load_learning_params(model_type,learning_params=dict(),DEBUG=False):
    
    if model_type=='VarNet':
        epochs=3 if DEBUG else 100
        loss=custom_loss_ssim
        metrics=[custom_metric_ssim,custom_loss_mse]
    else:
        epochs=3 if DEBUG else 200        
        loss=tfmri.losses.StructuralSimilarityLoss(image_dims=2)
        metrics=[tfmri.metrics.PeakSignalToNoiseRatio(image_dims=2),
                   tfmri.metrics.StructuralSimilarity(image_dims=2)]
        
    default_learning_params={'learning_rate': 10**-4,
                             'loss':loss,
                             'epochs':epochs,
                             'metrics':metrics,
                             }
    learning_params={**default_learning_params,**learning_params}
    learning_params['optimizer']=tf.keras.optimizers.Adam(learning_rate=learning_params['learning_rate'],clipnorm=1)
    return learning_params

# Load model
def load_models(inputs,model_type,config_model=dict(),DEBUG=False):
    #Load VarNets
    temp_config=default_configs(model_type,DEBUG=DEBUG)
    #Override default parameters with inputted parameters
    config_model={**temp_config,**config_model} 
    if model_type=='VarNet':
        #Load Model 1
        model=VarNet(**config_model)
        _ = model(inputs)

    elif model_type == '3DUNet':

        #Define and compile Model
        shape_input=(None,inputs.shape[-3],inputs.shape[-2],inputs.shape[-1])
        image_inputs= tf.keras.Input(shape_input)
        outputs=tfmri.models.UNet3D(**config_model)(image_inputs)
        model=tf.keras.Model(inputs=image_inputs,outputs=outputs) 
    
    elif model_type == 'FastDVDNet':

        #Define and compile Models
        shape_input=(inputs.shape[-3],inputs.shape[-2],5)
        image_inputs= tf.keras.Input(shape_input)
        outputs=FastDVDNet(**config_model)(image_inputs)
        model=tf.keras.Model(inputs=image_inputs,outputs=outputs)
        
    else:
        raise ValueError("model_type Unsupported")
   
    return model
