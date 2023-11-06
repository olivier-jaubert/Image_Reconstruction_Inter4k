import tensorflow as tf
import tensorflow_mri as tfmr
import tensorflow_nufft as tfft
rg = tf.random.Generator.from_seed(1, alg='philox')

# Default config
def config_base_preproc():
    config={'base_resolution': 256,
            'phases': 32,
            'roll': 0,
            'input_format': 'coil_compressed',
            'output_format': 'abs',
            'normalize_input': False,
    }
    return config

# Main Preprocessing Function
def preprocessing_fn(base_resolution=128,
                      phases=20,roll=0,output_format=None,input_format=None,gfilt=None,normalize_input=True,return_kspace=False):
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

      # Make fully sampled multicoil image.
      image = make_fs_rtcine_image(kspace, image_shape, roll=roll, phases=phases,keep_external_signal=False)

      npixel=tf.cast(base_resolution*base_resolution,tf.complex64)
      uskspace = tfft.nufft(tf.cast(image, tf.complex64), tf.expand_dims(traj, axis=-3),
                          transform_type='type_2',
                          fft_direction='forward')/tf.sqrt(npixel)

      # Apply density compensation.
      dcw = tfmr.flatten_density(dcw)[:phases,...]
      uskspace *= tf.cast(tf.expand_dims(dcw, axis=-2), tf.complex64)
      #Apply coil compression
      if input_format is not None and input_format=='coil_compressed':
        Coil_compressor = tfmr.coils.CoilCompressorSVD(coil_axis=-3,out_coils=10)
        #print(uskspace.shape,kspace.shape)
        Coil_compressor.fit(tf.expand_dims(uskspace[:,...],axis=-1))
        kspace=Coil_compressor.transform(kspace)
        uskspace=Coil_compressor.transform(tf.expand_dims(uskspace,axis=-1))[...,0]
      # Make fully sampled multicoil image from coil compressed Data
      image = make_fs_rtcine_image(kspace, image_shape, roll=roll, phases=phases,keep_external_signal=False)
      #print(image.shape,kspace.shape)

      # Combine coils and normalize ground truth images.
      ccimage = tfmr.coils.combine_coils(image, coil_axis=-3) 
      scalingfactor=tf.cast(tf.reduce_max(tf.abs(ccimage)),image.dtype)
      image=image/scalingfactor
      uskspace=uskspace/scalingfactor

      # Convert back to image space to get zero-filled image.
      zfill = tfft.nufft(uskspace, tf.expand_dims(traj, axis=-3),
                        grid_shape=image_shape,
                        transform_type='type_1',
                        fft_direction='backward')/tf.sqrt(npixel)

      # Combine coils and normalize ground truth images.
      image = tfmr.coils.combine_coils(image, coil_axis=-3) 
      # Combine coils and normalize zero-filled image.
      #zfill = tfmr.coils.combine_coils(zfill , coil_axis=-3) 
      if normalize_input:
        zfill = tfmr.scale_by_min_max(zfill)  # range [0, 1]
      
      zfill=tf.transpose(zfill,[2,3,0,1]) #height width time coils
      if gfilt is not None:
        image=tf.expand_dims(tf.complex(gaussian_blur(tf.math.real(image[0,...]),sigma=gfilt),gaussian_blur(tf.math.imag(image[0,...]),sigma=gfilt)),axis=0)
      image=tf.transpose(image,[1,2,0]) #height width time 

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
    if return_kspace:
      zfill={'zfill':zfill,'kspace':uskspace,'traj':traj,'dcw':dcw}
    return zfill, image  # input (height, width, time, channels=coils*2), output (height, width, time, channels=1 or 2 (mag or cpx))
  
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
    _body = lambda x: [tf.concat([x, x], axis=-4)]
    image = tf.while_loop(_cond, _body, [image],shape_invariants=[tf.TensorShape([None, None,None,None])])[0]
    image = image[:phases, ...]

    # `image` is now a fully-sampled multicoil multi-phase image.
    image = tfmr.resize_with_crop_or_pad(image, image_shape)
    return image

  def gaussian_blur(img, kernel_size=11, sigma=0.5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(tf.expand_dims(img,axis=-1))[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]
    outimage=tf.nn.depthwise_conv2d(tf.expand_dims(img,axis=-1), gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')
    return outimage[...,0]

  return _preprocessing_fn

def rolling_fn(phases=24,roll=1,rotation=0,input_format= None):
  """Returns a preprocessing function for training."""
  
  def _preprocessing_fn(zfill,image):
    """Preprocess the data.
    Roll and select nphases from zero filled images and image $selected_image2 (last image for lowest latency) 
    Args:
      inputs: Input data. A dict containing the following keys:
        - 'zfill': Zero filled images A tensor of shape [height, width,time (,channels)].
        - 'image': Ground truth images A tensor of shape [height, width,time (,channels)]
    Returns:
      A tuple (zerofilled image, ground truth image).
    """
    
    if tf.rank(zfill)==3:
      zfill=tf.expand_dims(zfill,axis=-1)
    zfill=tf.transpose(zfill,perm=[2,0,1,3])
    image=tf.transpose(tf.expand_dims(image,axis=-1),perm=[2,0,1,3])
    
    if rotation>0:
      rot_im=rg.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
      image=tf.image.rot90(image,k=rot_im)
      zfill=tf.image.rot90(zfill,k=rot_im)
      
    if roll>0:
        shift_im=rg.uniform(shape=(), minval=0, maxval=tf.shape(image)[0]-phases, dtype=tf.int32)
        image=tf.roll(image,shift=-shift_im,axis=0)
        zfill=tf.roll(zfill,shift=-shift_im,axis=0)
    image=image[:phases,...]
    zfill=zfill[:phases,...]

    #get rid of nans
    image = tf.where(tf.math.is_nan(image), 0., image)
    zfill = tf.where(tf.math.is_nan(zfill), 0., zfill)

    return zfill, image  # input (features), output (labels)
      
  return _preprocessing_fn