import os
import tqdm
import glob
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import functools
# Local imports (works if you are in project folder)
import utils.preprocessing_natural_videos as preprocessing_natural_videos
import utils.preprocessing_trajectory_gen as preproc_traj
import utils.preprocessing_fastdvdnet_noselect as preproc_fastdvdnet
import utils.preprocessing_rolling_fastdvdnet as preproc_fastdvdnet_roll

import utils.preprocessing_cartesian_VarNet as preproc_varnet
import utils.display_function as display_func
from utils.subsample_cartesian import generate_mask

import random

#Util Functions for Natural Images
def decode_videos(videofile):
  video = tf.io.read_file(videofile)
  video = tfio.experimental.ffmpeg.decode_video(video,0)
  video=video[:50,...]
  return {'video':video}

def resize_video(video,size=[244*2,244*6]):
  video=video['video']
  video=tf.image.resize(video,size, method='bilinear', preserve_aspect_ratio=True, antialias=False)
  video=tf.cast(tf.clip_by_value(video,0,255),tf.uint8)
  return {'video':tf.ensure_shape(video,(None,)*4)}

def split_training_test_set(folder_name,n=692,split=[0.75,0.10,0.15],verbose=1):
    """
    folder_name: path to dataset usually config['data_folder']
    n: takes n elements for the dataset (-1 for whole dataset)
    split: [train, val, test] proportion of the n elements attributed to training, validation and testing
    verbose: print the split
    """
    # Read files and split data 
    train_files=[]
    val_files=[]
    test_files=[]

    filelist=[]
    path_h5_data=os.path.join(folder_name,'*')
    for file in tqdm.tqdm(glob.glob(path_h5_data)):
        filelist.append(file)        
    sorted_files = sorted(filelist,key=lambda x: int(os.path.splitext(os.path.split(x)[-1])[-2]))
    sorted_files = [x for x in sorted_files] 
    list_to_remove=['312.mp4','66.mp4','675.mp4','502.mp4',]
    for bad_element in list_to_remove:
        try:
            sorted_files.remove(os.path.join(os.path.split(path_h5_data)[0],bad_element))#these videos have frames including only zeros leading to nans
        except: print(os.path.join(os.path.split(path_h5_data)[0],bad_element),':file not in list')
    if n==-1:
        n=len(sorted_files)

    ntrain=int(split[0]*n)
    nval=int(split[1]*n)
    ntest=int(np.ceil(split[2]*n))
    #print('leftovers:',n-ntrain-nval-ntest)
    train_files.append(sorted_files[:ntrain])
    val_files.append(sorted_files[ntrain:ntrain+nval])
    test_files.append(sorted_files[ntrain+nval:ntrain+nval+ntest])

    train_files=np.concatenate(train_files,axis=0)
    val_files=np.concatenate(val_files,axis=0)
    test_files=np.concatenate(test_files,axis=0)

    # Shuffle files.
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    if verbose:
        print('Total/Train/Val/Test:',len(train_files)+len(val_files)+len(test_files),
            '/',len(train_files),'/',len(val_files),'/',len(test_files),'leftovers:',n-ntrain-nval-ntest)
    return [train_files,val_files,test_files]


def generate_dataset_VarNet(datasets,dataset_dir,config_natural_images=None,config_preproc=None,config_traj=None,reset=1,DEBUG=False):
    """
    Inputs:
    datasets: list of files [train_files,val_files,test_files]
    dataset_dir: path to cached data
    reset: 0,1 or 2 -> 0 resuses cached data if exists / 1 cleans previously cached data after Resizing of Inter4k Data / 2 cleans all previously cached data  
    Returns:
    datasets_withtransforms: tf.Data.Dataset objects with transformation from videos to inputs to FastDVDnet
    """
    if config_preproc is None:
        config_preproc=preproc_varnet.config_base_preproc()
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],config_preproc['phases'])
              
    if config_traj is None:
        config_traj={'acc':14,
        'phases':15,
        'center_lines':8,
        'half_fourier':0.6,
        'mask_type':'noreselect'}
    

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)
    
    #Define Preprocessing from natural image to kspace to network inputs
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    #Init functions:
    video=decode_videos(datasets[0][0])
    video=resize_video(video,size=[244*2,244*6])
    _=preproc_natural_image(video)
    seed_value=7
    mask=generate_mask(seed_value,(config_traj['phases'],1,config_preproc['base_resolution'],config_preproc['base_resolution'],1),[config_traj['acc']],[config_traj['center_lines']],half_fourier=config_traj['half_fourier'],mask_type=config_traj['mask_type'])     
    preproc_function=preproc_varnet.preprocessing_fn(**config_preproc)

    cache_initial=dataset_dir+'/cache_Inter4K_preproc/trainv3/'
    cache_initial_val=dataset_dir+'/cache_Inter4K_preproc/valv3/'
    cache_initial_test=dataset_dir+'/cache_Inter4K_preproc/testv3/'
    # Reperform resizing of Inter4K Dataset (a bit long)
    if reset==2:
        shutil.rmtree(cache_initial, ignore_errors=True, onerror=None)
        shutil.rmtree(cache_initial_val, ignore_errors=True, onerror=None)
        shutil.rmtree(cache_initial_test, ignore_errors=True, onerror=None)
    os.makedirs(cache_initial,exist_ok=True)
    os.makedirs(cache_initial_val,exist_ok=True)
    os.makedirs(cache_initial_test,exist_ok=True)
    cache_folder=dataset_dir+'/cache_inter4k_varnet_v2/'
    cache_folder2=dataset_dir+'/cache_inter4k_varnet_val_v2/'
    if reset>=1:
        shutil.rmtree(cache_folder, ignore_errors=True, onerror=None)
        shutil.rmtree(cache_folder2, ignore_errors=True, onerror=None)
    os.makedirs(cache_folder,exist_ok=True)
    os.makedirs(cache_folder2,exist_ok=True)

    dataset_withtransforms=[]
    for pp,dataset in enumerate(datasets):
        print(pp)
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(list(map(str, dataset)), dtype=tf.string))
        dataset=dataset.map(decode_videos,num_parallel_calls=1)
        dataset=dataset.map(functools.partial(resize_video,size=[244*2,244*6]),num_parallel_calls=1)
        if pp==0:
            dataset=dataset.cache(cache_initial)
        elif pp==1:
            dataset=dataset.cache(cache_initial_val)
        elif pp==2:
            dataset=dataset.cache(cache_initial_test)
        
        dataset=dataset.map(preproc_natural_image,num_parallel_calls=1)

        dataset_traj=tf.data.Dataset.from_tensors({'mask':mask}).repeat(-1)
        ds= tf.data.Dataset.zip((dataset,dataset_traj))
        dataset = ds.map(lambda image, mask: {'image': image, 'trajectory': mask})
        dataset=dataset.map(preproc_function,num_parallel_calls=1)
        if pp==0:
            dataset=dataset.cache(cache_folder)
        if not DEBUG:
            dataset=dataset.shuffle(buffer_size=4,seed=1)
        if pp==1:
            dataset=dataset.cache(cache_folder2)

        dataset=dataset.batch(1,drop_remainder=True)
        if not DEBUG:
            dataset=dataset.prefetch(4)
        dataset_withtransforms.append(dataset)
    
    return dataset_withtransforms




