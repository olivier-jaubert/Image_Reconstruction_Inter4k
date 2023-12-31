import tensorflow as tf
import tensorflow_mri as tfmr
import tensorflow_nufft as tfft

rg = tf.random.Generator.from_seed(1, alg='philox')

def config_base_preproc():
    config={'base_resolution': 240,
            'phases': 12,
            'roll': 0,
            'input_format': 'abs',
            'output_format': 'abs',
            'normalize_input': False,
    }
    return config

def preprocessing_fn(base_resolution=128,
                      phases=20,roll=0,output_format=None,input_format=None,normalize_input=True):
  """Returns a preprocessing function for training."""
  
  def _preprocessing_fn(inputs):
    """Preprocess the data.

    Takes a fully sampled image, resamples k-space onto an arbitrary trajectory and
    returns the zerofilled and the ground truth image.

    Args:
      inputs: Input data. A dict containing the following keys:
        - 'kspace': A tensor of shape [nslices,ncoils,time, height, width].
        - 'traj': A dictionary containing 'traj': trajectory [time,nspirals,nreadout,xypos]

    Returns:
      A tuple (zerofilled image, ground truth image).
    """
    with tf.device('/gpu:0'):
      shape=tf.shape(inputs['image']['kspace'])

      kspace = inputs['image']['kspace'][shape[0]//2,:,:,...]
      
      image_shape = [base_resolution, base_resolution]
      # Make fully sampled multicoil image.
      image = make_fs_rtcine_image(kspace, image_shape, roll=roll, phases=phases,keep_external_signal=False)

      # Combine coils and normalize ground truth images.
      ccimage = tfmr.coils.combine_coils(image, coil_axis=-3) 
      scalingfactor=tf.cast(tf.reduce_max(tf.abs(ccimage)),image.dtype)
      image=image/scalingfactor
      kspace=kspace/scalingfactor
      
      traj= inputs['traj']['traj']
      dcw= inputs['traj']['dcw']
      reps=tf.cast(tf.math.ceil(phases/tf.shape(traj)[0]), tf.int32)
      traj=tf.tile(traj,[reps,1,1,1])
      dcw=tf.tile(dcw,[reps,1,1])
      
      traj = tfmr.flatten_trajectory(traj)
      if roll>0:
        shift=rg.uniform(shape=(), minval=0, maxval=tf.shape(traj)[0], dtype=tf.int32)
        traj=tf.roll(traj,shift=shift,axis=0)
      traj=traj[:phases,...]

      npixel=tf.cast(base_resolution*base_resolution,tf.complex64)
      kspace = tfft.nufft(tf.cast(image, tf.complex64), tf.expand_dims(traj, axis=-3),
                          transform_type='type_2',
                          fft_direction='forward')/tf.sqrt(npixel)

      # Apply density compensation.
      dcw = tfmr.flatten_density(dcw)[:phases,...]
      kspace *= tf.cast(tf.expand_dims(dcw, axis=-2), tf.complex64)
      # Convert back to image space to get zero-filled image.
      zfill = tfft.nufft(kspace, tf.expand_dims(traj, axis=-3),
                        grid_shape=image_shape,
                        transform_type='type_1',
                        fft_direction='backward')/tf.sqrt(npixel)

      # Combine coils and normalize ground truth images.
      image = tfmr.coils.combine_coils(image, coil_axis=-3) 
      # Combine coils and normalize zero-filled image.
      zfill = tfmr.coils.combine_coils(zfill , coil_axis=-3) 
      if normalize_input:
        zfill = tfmr.scale_by_min_max(zfill)  # range [0, 1]
      
      zfill=tf.transpose(zfill,[1,2,0]) #height width time (time as channels)
      image=tf.transpose(image,[1,2,0]) #height width time (time as channels)

      if input_format is not None and input_format=='abs':
        zfill=tf.math.abs(zfill)
      else:
        zfill = tf.concat((tf.math.real(zfill),tf.math.imag(zfill)), axis=-1)

      if output_format is not None and output_format=='abs':
        image=tf.math.abs(image)
      elif output_format is not None and output_format=='abspre':
        image=tf.math.abs(image)
      else:
        image = tf.concat((tf.math.real(image),tf.math.imag(image)), axis=-1)
        
    return zfill, image  # input (features), output (labels)
  
  def make_fs_rtcine_image(kspace, image_shape, roll=0,phases=20,  keep_external_signal=False):
    """Returns a fully sampled image from k-space."""
    kspace=tf.ensure_shape(kspace,(None,) * 4)
    # Crop to fixed size and normalize pixel intensities (in image space).
    image = tfmr.signal.ifft(kspace, axes=[-2, -1], norm='ortho', shift=True)
    if not keep_external_signal:
      image = tfmr.resize_with_crop_or_pad(image, image_shape)

    # Select random starting point.
    input_phases = tf.shape(image)[-4]
    if roll>0:
      random_shift = rg.uniform(
          (), minval=0, maxval=input_phases, dtype=tf.int32)
      image = tf.roll(image, shift=random_shift, axis=-4)

    # Pad up to specified number of phases.
    _cond = lambda x: tf.math.less(tf.shape(x)[-4], phases)
    _body = lambda x: tf.concat([x, x], axis=-4)
    image = tf.while_loop(_cond, _body, [image],shape_invariants=[tf.TensorShape([None, None,None,None])])[0]
    image = image[:phases, ...]

    # `image` is now a fully-sampled multicoil multi-phase image.
    image = tfmr.resize_with_crop_or_pad(image, image_shape)
    return image
  return _preprocessing_fn


def rolling_fn(phases=5,roll=1,selected_image2=-1):
  """Returns a preprocessing function for training."""
  
  def _preprocessing_fn(zfill,image):
    """Preprocess the data.
    Roll and select nphases from zero filled images and image $selected_image2 (last image for lowest latency) 
    Args:
      inputs: Input data. A dict containing the following keys:
        - 'zfill': Zero filled images A tensor of shape [height, width,time].
        - 'image': Ground truth images A tensor of shape [height, width,time]
    Returns:
      A tuple (zerofilled image, ground truth image).
    """
    if roll>0:
        shift_im=rg.uniform(shape=(), minval=0, maxval=tf.shape(image)[-1]-phases, dtype=tf.int32)
        image=tf.roll(image,shift=-shift_im,axis=-1)
        zfill=tf.roll(zfill,shift=-shift_im,axis=-1)
      
    image=image[...,:phases]
    zfill=zfill[...,:phases]
    
    if selected_image2 is not None: 
      image=tf.expand_dims(image[...,selected_image2],axis=-1)
    #get rid of nans
    image = tf.where(tf.math.is_nan(image), 0., image)
    zfill = tf.where(tf.math.is_nan(zfill), 0., zfill)
    return zfill, image  # input (features), output (labels)
      
  return _preprocessing_fn
