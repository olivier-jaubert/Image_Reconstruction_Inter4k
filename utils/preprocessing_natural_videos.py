import tensorflow as tf
import tensorflow_mri as tfmri
import tensorflow_addons as tfa
import math
import numpy as np
import scipy.signal
import tf_clahe
rg = tf.random.Generator.from_seed(1, alg='philox')

def preprocessing_fn(base_resolution=128,num_coils=10, masking=False, return_gt=False, complex_transform=0,phases=None,regsnr=0,sigma_coil=4, addmotion=0,apply_filter=0,kernel_size=0,add_phase=0,image_adjustment=[0,1],clahe=False):
  """Returns a preprocessing function for training."""
  def _preprocessing_fn(tfds_video):
    """From jpg to formatted kspace of image.
    Takes natural videos and maps to magnitude only images.
    Args:
      tfds_video: output of tensorflow dataset loading DAVIS dataset (dictionary with tfds_video['metadata'] 
      and tfds_video['video']['frames'])
    Returns:
      Ground truth multi coil kspace.
    """
    with tf.device('/gpu:0'):
      #Load data
      if isinstance(tfds_video['video'], dict):
        image_series=tf.cast(tfds_video['video']['frames'],tf.float32)
      else:
          image_series=tf.cast(tfds_video['video'],tf.float32)
      #Crop Video
      nonlocal phases
      if phases is None:
        phases=tf.shape(image_series[0])
      image_series=tfmri.resize_with_crop_or_pad(image_series,shape=[phases,base_resolution,base_resolution,-1])
      if clahe:
        #print(image_series.shape)
        image_series=tf_clahe.clahe(image_series, gpu_optimized=True)
        #image_series=fast_clahe(image_series)
      #image_series=tf.cast(image_series,tf.float32)
      image_series=tfmri.scale_by_min_max(image_series)
      #Contrast adjustments
      nonlocal image_adjustment
      image_adjustment=tf.cast(image_adjustment,image_series.dtype)
      image_series=image_adjustment[0]+image_adjustment[1]*image_series
      image_series=tf.clip_by_value(image_series, 0, 1)
      #Take 2 random RGB channels and scale relative phase between the two to create complex image
      if complex_transform>0:
          shift_cha=rg.uniform(shape=(), minval=0, maxval=tf.shape(image_series)[-1], dtype=tf.int32)
          image_series=tf.roll(image_series,shift=shift_cha,axis=-1)
          image_series=tfmri.resize_with_crop_or_pad(image_series,shape=[phases,base_resolution,base_resolution,2])
          image_series=tf.complex(image_series[:,:,:,0],image_series[:,:,:,1])
          image_series=tf.cast(tf.math.abs(image_series),tf.complex64)*tf.math.exp(1j*tf.cast(tf.math.angle(image_series)*complex_transform,tf.complex64))
      else:
          image_series=tf.math.sqrt(tf.reduce_sum(tf.cast(image_series,tf.float32)**2,axis=-1))
          image_series=tf.cast(image_series,tf.complex64)
      #apply elliptical masking
      if masking:
          mask_shape=[base_resolution,base_resolution]
          mask=tf.py_function(elliptical_mask, inp=[mask_shape],Tout=tf.uint8)
          mask.set_shape(mask_shape)
          image_series=image_series*tf.expand_dims(tf.cast(mask,image_series.dtype)+10**-8,axis=0)

      # Simulate coil sensitivity maps
      smaps=simulate_coils(base_resolution,sigma_coil,num_coils,coil_size=base_resolution*2,add_phase=add_phase,ngrid=2)
      if masking:
        smaps=smaps*tf.expand_dims(tf.cast(mask,image_series.dtype),axis=0)
      #Apply Filter on the corners
      nonlocal kernel_size
      if kernel_size==0:
            kernel_size=base_resolution
      if apply_filter>0:
        temp=tukey_kernel(1,kernel_size,sigma=apply_filter)
        temp=tfmri.resize_with_crop_or_pad(temp,[base_resolution,base_resolution,1])
        image_series=image_series*tf.cast(tf.expand_dims(temp[...,0],axis=0),dtype=tf.complex64)
      #Add background phase to the object
      if add_phase>0:
          image_series=_backgroundphase(image_series)
      
      #Create Coil Images from object and sensitivity maps
      coil_images=tf.expand_dims(smaps,axis=1)*tf.expand_dims(image_series,axis=0)
      sos_image=tf.abs(tf.sqrt(tf.reduce_sum(coil_images*tf.math.conj(coil_images),axis=0)))
      coil_images=coil_images/tf.cast(tf.reduce_max(sos_image),tf.complex64)
      
      #Add independant white gaussian noise to each coil image.
      if len(regsnr)==2:
            tempregsnr=rg.uniform(shape=(), minval=regsnr[0], maxval=regsnr[1], dtype=tf.float32)
      else:
            tempregsnr=regsnr[0]
      if tempregsnr>0:
        coil_images=_awgn(coil_images,tempregsnr,cpx=True)
      #Create ground truth kspace and sos ground truth image
      #sos_image=sos_image/tf.reduce_max(sos_image)
      kspace=tfmri.signal.fft(coil_images, axes=[-2, -1], norm='ortho', shift=True)
      kspace=tf.cast(kspace,tf.complex64)
      kspace=tf.transpose(kspace,[1,0,2,3])
      
      # Dictionary with cartesian multi-coil kspace data in 'kspace' field
      #Format [slice, phases, coils, x, y]
      output=dict()
      output['kspace']=tf.expand_dims(kspace,axis=0)
      
    if return_gt:
        output['gt']=image_series
        output['sensitivities']=smaps
    return output

  return _preprocessing_fn


def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        axy = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        #kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = tf.exp(-xx ** 2  / (2.0 * sigma[0] ** 2)- yy ** 2/ (2.0 * sigma[1] ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

def _backgroundphase(image, ngrid=6,scaling=1):
    #Adds random background phase to already complex images Fixed along 1st (time) dimension
    siz=(ngrid,)*2
    arr = rg.uniform(siz)*2*scaling#*math.pi
    #arr = zoom(arr, size[0]/6)
    arr=tf.expand_dims(tf.expand_dims(arr,axis=0),axis=-1)
    arr=tf.image.resize( arr,  image.shape[1:],'bicubic')
    #arr=tf.image.crop_and_resize(arr,[[0,0,5,5]],[0],image.shape[1:])
    arr=arr[...,0]
    #arr=skimage.transform.resize(arr, image.shape[1:],order=3)
    randphase=(arr + rg.uniform((1,))*2*math.pi)
    image= image*tf.exp(1j*tf.cast(randphase,tf.complex64))
    return image

def simulate_coils(base_resolution,sigma_coil,num_coils,coil_size=None,add_phase=0,ngrid=6):
      if coil_size is None:
           coil_size=base_resolution
      # Simulate coil sensitivity maps
      smaps = []
      if len(num_coils)==2:
            total_coils=rg.uniform(shape=(), minval=num_coils[0], maxval=num_coils[1], dtype=tf.int32)
            nsim=num_coils[1]
      else:
            total_coils=num_coils[0]
            nsim=num_coils[0]
      #total_coils = num_coils
      
      #base =tf.clip_by_value(base/tf.reduce_max(base), 0.0, 0.3) # flat top sensitivity
      for coil in range(nsim):
          if len(sigma_coil)==2:
            sigma_x=rg.uniform(shape=(1,), minval=sigma_coil[0], maxval=sigma_coil[1])
            sigma_y=rg.uniform(shape=(1,), minval=sigma_coil[0], maxval=sigma_coil[1])
            sigmas=[base_resolution/sigma_x,base_resolution/sigma_y]
          else:
               sigmas=[base_resolution/sigma_coil[0],base_resolution/sigma_coil[0]]

          base = gauss_kernel(2,coil_size, sigma = sigmas)
          #random_phase=tf.cast(rg.uniform(shape=(1,), minval=0, maxval=math.pi),dtype=tf.complex64)
          random_phase=tf.cast(rg.uniform(shape=(1,), minval=-math.pi, maxval=math.pi),dtype=tf.complex64)
          random_intensity=tf.cast(rg.uniform(shape=(1,), minval=0.1, maxval=1),dtype=tf.complex64)
          # # coils in random positions (a bit away from the edges)
          # transform=tf.cast(rg.uniform(shape=(2,), minval=-base_resolution//3, maxval=base_resolution//3),dtype=tf.float32)
          # coils in random positions (excluding center)
          transform=tf.cast(rg.uniform(shape=(2,), minval=base_resolution//5, maxval=base_resolution//2),dtype=tf.float32)
          random_locations=tf.cast(rg.uniform(shape=(2,),minval=-1,maxval=9,dtype=tf.int32),dtype=tf.float32)
          transform=transform*[(-1)**random_locations[0],(-1)**random_locations[1]]

          temp=tfa.image.translate(base,transform,'bilinear' )#+0.01
          temp=random_intensity*tf.complex(temp[:,:,0],temp[:,:,1])*tf.exp(1j*random_phase)
          temp=tfmri.resize_with_crop_or_pad(temp,[base_resolution,base_resolution])
          #add random phase background to each coil
          if add_phase>0:
            temp=_backgroundphase(tf.expand_dims(temp,axis=0),ngrid=ngrid,scaling=add_phase)[0,...]
          smaps.append(temp)
      axis=0
      idxs = tf.range(tf.shape(smaps)[axis])
      ridxs = tf.random.shuffle(idxs)[:total_coils]
      smaps = tf.gather(smaps, ridxs,axis=axis)
      smaps=tf.stack(smaps,axis=0)
      sos_smaps=tf.abs(tf.sqrt(tf.reduce_sum(smaps*tf.math.conj(smaps),axis=0)))
      #smaps=tf.complex(smaps[:,:,:,0],smaps[:,:,:,1])
      #sos_smaps=tf.cast(sos_smaps,tf.complex64)
      smaps = smaps / tf.cast(tf.reduce_max(sos_smaps),tf.complex64)
      return smaps

def tukey_kernel(channels, kernel_size, sigma=0.5):
        window = scipy.signal.windows.tukey(kernel_size,sigma)
        window1d = tf.abs(window)
        kernel = tf.sqrt(tf.tensordot(window1d,window1d,axes=0))
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel



#@tf.function
def _geometric_augmentation(image,phases,motion_proba=0.5,motion_ampli=0.8,maxrot=45.0):
  transform=tf.zeros((tf.shape(image)[0],2))
  image_size=tf.shape(image)
  if rg.uniform(shape=())<motion_proba:
    displacement=[]
    for idxdisp in range(phases):
        if idxdisp%5==0:
            transform=rg.uniform(shape=(2,1), minval=-motion_ampli, maxval=motion_ampli)
        else:
            transform=transform
        if idxdisp==0:
            displacement.append(tf.cast([[0], [0]],tf.float32))
        else:
            displacement.append(transform+displacement[idxdisp-1])

    transform=tf.stack(displacement)
    transform=tf.squeeze(transform)
    #print(image.shape,transform.shape,transform.dtype)
    image=tfa.image.translate(image,transform,'bilinear' )

  if maxrot>0:
    #ROTATION
    rotationangle=rg.uniform(shape=(),minval=-maxrot,maxval=maxrot)
    
    image=tfa.image.rotate(image, angles=rotationangle,interpolation='bilinear')
    cpx_size=tf.shape(image)
    image=image[:,cpx_size[1]//2-image_size[1]//2:(cpx_size[1]//2-image_size[1]//2+image_size[1]),
            cpx_size[2]//2-image_size[2]//2:(cpx_size[2]//2-image_size[2]//2+image_size[2]),...]
  return image

def _awgn(data, regsnr,cpx=False): 
    """
    Add White Gaussian noise to reach target snr if regsnr>1 else apply noise with stddev = regsnr
    """
    if regsnr>1:
      sigpower = tf.reduce_mean(tf.abs(data) ** 2)
      noisepower = sigpower / (10 ** (regsnr / 10))
    else:
      noisepower=regsnr**2
    if cpx:
      noise = tf.complex(rg.normal(shape=tf.shape(data), mean=0.0, stddev=tf.math.sqrt(noisepower)),rg.normal(shape=tf.shape(data), mean=0.0, stddev=tf.math.sqrt(noisepower)))
    else:
      noise = rg.normal(shape=tf.shape(data), mean=0.0, stddev=tf.math.sqrt(noisepower))
    data += noise
    return data

def elliptical_mask(shape):
  import cv2
  # Window name in which image is displayed
  # Color in BGR
  color = (1, 1, 1)
  shape=shape.numpy()
  mask=np.zeros(np.concatenate((shape,[3,])),dtype=np.int32)
  center_coordinates = (shape[-2]//2, shape[-1]//2)
  axesLength = (int(shape[-2]//2*rg.uniform(shape=(),minval=1.0, maxval=1.4)), int(shape[-1]//2*0.8*rg.uniform(shape=(),minval=0.8, maxval=1.2)))
  #axesLength = (int(shape[-2]//2*1.4), int(shape[-1]//2))
  angle = int(rg.uniform(shape=(),minval=0, maxval=360))
  startAngle = 0
  endAngle = 360
  # Line thickness of -1 px (fills the ellipse)
  thickness = -1
  #print(mask.shape,axesLength)
  # Using cv2.ellipse() method
  # Draw a ellipse with blue line borders of thickness of -1 px
  mask = cv2.ellipse(mask, center_coordinates, axesLength, angle,
                          startAngle, endAngle, color, thickness)
  # # Create an elliptic hole within the ellipse
  # mask2=np.zeros(np.concatenate((shape,[3,])),dtype=np.int32)
  # center_coordinates2 = (int(shape[-2]//2*(1+rg.uniform(shape=(),minval=-0.8, maxval=0.8))), int(shape[-1]//2*(1+rg.uniform(shape=(),minval=-0.8, maxval=0.8))))
  # hole_size_factor = 4
  # axesLength2 = (int(shape[-2]//2*rg.uniform(shape=(),minval=1.0, maxval=1.4))//hole_size_factor, 
  #                int(shape[-1]//2*0.8*rg.uniform(shape=(),minval=0.8, maxval=1.2))//hole_size_factor)
  
  # mask2 = cv2.ellipse(mask2, center_coordinates2, axesLength2, angle,
  #                         startAngle, endAngle, color, thickness)
  # mask[mask2>0]=0
  return mask[:,:,0]

@tf.function(experimental_compile=True)  # Enable XLA
def fast_clahe(img):
    return tf_clahe.clahe(img, gpu_optimized=True)