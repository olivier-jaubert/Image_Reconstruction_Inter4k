#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday October 10 10:11:42 2023
Example Code for training an MRI Image Reconstruction Network from Inter4k Dataset

Three networks and trajectories were investigated: 
- Cartesian real-time with unrolled VarNet [ref] -> model_type='VarNet'
- Radial real-time with multicoil 3DUNet [ref] -> model_type='3DUNet'
- Low latency spiral imaging with FastDVDNet (Hyperslice [ref]) -> model_type='FastDVDNet'

Provided code includes trajectories, model training and pre-trained models as implemented for the paper.

Methods details in : 
Investigating the use of high spatio-temporal resolution publicly available natural videos to learn Dynamic MR image reconstruction

@author: Dr. Olivier Jaubert
"""

#For python command line call
import argparse
parser = argparse.ArgumentParser(prog='Train a model using arguments model_type')
parser.add_argument('-m', '--model_type',default='FastDVDNet')
parser.add_argument('-d', '--debug', action='store_true')
args=parser.parse_args()
#Choose model_type
model_type=args.model_type
DEBUG=args.debug
result_folder='TrainingFolder'
print('Training with the following parameters:',args)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import numpy as np
import tensorflow as tf
try:  tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
except: print('running on CPU')
import tensorflow_mri as tfmri
import random
import matplotlib.pyplot as plt
import datetime
import json
# # Local imports (works if you are in project folder)
import utils.create_tensorflow_dataset as dataset_utils
import utils.model_utils as model_utils
import utils.display_function as display_func

#Set seed for all packages
seed_value=1
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
("")

# Dataset and preprocessings 
exp_name='FullInter4k_'+model_type if not DEBUG else 'Default_'+model_type+'_DEBUG'
data_folder = './DatasetFolder/Inter4K/60fps/UHD/'
cache_dir = './DatasetFolder/Full/DEBUG/' if DEBUG else './DatasetFolder/Full/'

#Total number of samples from dataset used 
n= 12 if DEBUG else -1

#Reset parameter: 
#2 for full repreprocessing from original Inter4K.
#1 from preprocessed resized video (recommended).
#0 from cached data from previous run.
reset=2

#Selecting files and splitting train/val/test 
filenames_datasets=dataset_utils.split_training_test_set(data_folder,n=n,split=[0.945,0.05,0.005],verbose=1)
#Resizing Original Inter4K videos
preproc_datasets=dataset_utils.run_load_preproc_dataset(filenames_datasets,cache_dir,reset=reset)
#Creating Datasets Corresponding to Acquisition and reconstruction method
if model_type=='VarNet':
    dataset_withtransforms=dataset_utils.generate_dataset_VarNet(preproc_datasets,cache_dir,reset=reset,DEBUG=DEBUG)
elif model_type=='3DUNet':
    dataset_withtransforms=dataset_utils.generate_dataset_radial3DUNet(preproc_datasets,cache_dir,reset=reset,DEBUG=DEBUG)
elif model_type=='FastDVDNet':
    dataset_withtransforms=dataset_utils.generate_dataset_FastDVDNet(preproc_datasets,cache_dir,reset=reset,DEBUG=DEBUG)
else:
    print('model_type: ',model_type,' not supported')

#Preprocessing 1 input to get shapes.
inputs_temp=next(iter(dataset_withtransforms[0].take(1)))

config_model=dict()
#To modify model architecture:
# config_model={'filters': [16,32,46,64]}

learning_params=dict()
#To modify learning params :
# learning_params={'learning_rate': 0.005}

model=model_utils.load_models(inputs_temp,model_type,DEBUG=DEBUG)
learning_params=model_utils.load_learning_params(model_type,DEBUG=DEBUG,learning_params=learning_params)
print(learning_params)
model.compile(optimizer=learning_params['optimizer'],
                    loss=learning_params['loss'],
                    metrics=learning_params['metrics'],
                    run_eagerly=False)
model.summary()

#Defining Paths
exp_name += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(result_folder, exp_name)

callbacks=[]
checkpoint_filepath=os.path.join(exp_dir,'ckpt/saved_model')
callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
        save_best_only=True))
callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_dir,'logs')))

#Train Model
history=model.fit(dataset_withtransforms[0],
          epochs=learning_params['epochs'],
          verbose=1,
          callbacks=callbacks,
          validation_data=dataset_withtransforms[1]
          )

fig = plt.figure(figsize=(16,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
leg = plt.legend()

#Evaluate On Test Set
checkpoint_filepath=os.path.join(exp_dir,'ckpt/saved_model')
model.load_weights(checkpoint_filepath)
result = model.evaluate(dataset_withtransforms[2])
if model_type=='VarNet':
    results_dict={model.metrics_names[0]: result} #when only 1 metric
else:
    results_dict=dict(zip(model.metrics_names, result))
filename = os.path.join(exp_dir,'results.json')
with open(filename, 'w') as f:
    f.write(json.dumps(results_dict))

#Inference
#Preproc series 1
if DEBUG:
    ntestimage=0
else:
    ntestimage=9
bg_color='k'
if model_type=='VarNet':
    inputs=next(iter(dataset_withtransforms[-1].skip(ntestimage)))
    output = model(inputs)
    zfill=tf.complex(inputs[0][0,...,0],inputs[0][0,...,1])
    gt=tf.complex(inputs[-1][0,...,0],inputs[-1][0,...,1])
    output=tf.complex(output[0,...,0],output[0,...,1])
elif model_type=='3DUNet':
    inputs,gt=next(iter(dataset_withtransforms[-1].skip(ntestimage)))
    output = model(inputs)
    zfill=tf.complex(inputs[0,...,:10],inputs[0,...,10:])
    zfill=np.sqrt(np.sum(zfill*np.conj(zfill),axis=-1))
    gt=gt[0,...,0]
    output=output[0,...,0]
elif model_type=='FastDVDNet':
    inputs,gt=next(iter(dataset_withtransforms[-1].skip(ntestimage)))
    output = model(inputs)
    zfill=inputs[0,...,-1]
    gt=gt[0,...,0]
    output=output[0,...,0]

#From Left to Right: Input, Ground Truth, Reconstructed.
if model_type=='FastDVDNet':
    savepath=os.path.join(exp_dir,'test_image_'+str(ntestimage)+'_'+model_type)
    plot_image=np.abs(np.concatenate((zfill,output,gt),axis=1))
    plt.figure(figsize=(15,5),facecolor=bg_color)
    plt.imshow(plot_image,vmin=0,vmax=1,cmap='gray')
    plt.axis('off')
    plt.savefig(savepath)
else:
    savepath=os.path.join(exp_dir,'test_video_'+str(ntestimage)+'_'+model_type)
    plot_image=np.abs(np.concatenate((zfill,output,gt),axis=2))
    display_func.plotVid(plot_image,axis=0,vmin=0,vmax=1,interval=41.66,savepath=savepath,bg_color=bg_color)

