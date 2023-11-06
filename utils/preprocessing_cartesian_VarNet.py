import tensorflow as tf
import tensorflow_mri as tfmri
import tensorflow_nufft as tfft

rg = tf.random.Generator.from_seed(1, alg='philox')

## Default config
def config_base_preproc():
    config={'base_resolution': 240,
            'phases': 24,
            'roll': 0,
            'normalize_input': False,
            'input_format': 'coil_compressed',
    }
    return config

## Main Preprocessing Function
def preprocessing_fn(base_resolution=128,
                      phases=20,roll=0,normalize_input=False,input_format='coil_compressed'):
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
      mask=tf.cast(inputs['trajectory']['mask'][...,0],kspace.dtype)
      #trajphases=tf.shape(mask)[0]
      image_shape = [base_resolution, base_resolution]
      # Prepare fully sampled multicoil image.
      image = make_fs_rtcine_image(kspace, image_shape, roll=roll, phases=phases,keep_external_signal=False)
      kspace = tfmri.signal.fft(image, axes=[-2, -1], norm='ortho', shift=True)
      shape=tf.shape(kspace)
      if roll>0:
        shift=rg.uniform(shape=(), minval=0, maxval=tf.shape(mask)[0], dtype=tf.int32)
        mask=tf.roll(mask,shift=shift,axis=0)
      reps=tf.cast(tf.math.ceil(phases/tf.shape(mask)[0]), tf.int32)
     
      mask=tf.tile(mask,[reps,1,1,1])[:phases,...]
      #Mask kspace
      uskspace = kspace * mask + tf.cast(0.0,dtype=kspace.dtype)

      #Apply coil compression
      if input_format == 'coil_compressed':
        Coil_compressor = tfmri.coils.CoilCompressorSVD(coil_axis=-3,out_coils=10)
        #print(uskspace.shape,kspace.shape)
        Coil_compressor.fit(uskspace[:,...])
        kspace=Coil_compressor.transform(kspace)
        uskspace=Coil_compressor.transform(uskspace)
        # Make fully sampled multicoil image from coil compressed Data
        image = make_fs_rtcine_image(kspace, image_shape, roll=0, phases=phases,keep_external_signal=False)
        #print(image.shape,kspace.shape)

      #Estimate Coil sensitivities:
      filter_fn = lambda x: tfmri.signal.hann(7 * x)
      filtered_kspace = tfmri.signal.filter_kspace(tf.reduce_sum(uskspace,axis=0,keepdims=True)*tf.cast(tf.math.divide_no_nan(1.0,tf.reduce_sum(tf.cast(mask,dtype=tf.float32),axis=0,keepdims=True)),tf.complex64), filter_fn=filter_fn,filter_rank=2)
      filtered_kspace=filtered_kspace/tf.cast(tf.reduce_max(tf.abs(filtered_kspace )),dtype=tf.complex64)
      low_res_images = tfmri.recon.adj(filtered_kspace, image_shape)
      
      sensitivities = tfmri.coils.estimate_sensitivities(
                    low_res_images, coil_axis=-3, method='inati',
                    filter_size=9,max_iter=10,tol=10**-6)
      null_thresh=0.1
      sos=tf.abs(tf.reduce_sum(sensitivities*tf.math.conj(sensitivities),axis=-3,keepdims=True))
      masksens=tf.where(sos/tf.reduce_max(sos) < null_thresh, tf.cast(0.,tf.complex64), tf.cast(1.,tf.complex64))
      sensitivities *=masksens
      if tf.reduce_max(tf.abs(sensitivities))==0:
         sensitivities =sensitivities+1
         
      # Combine coils and normalize ground truth images.
      image = tfmri.coils.combine_coils(image, maps=sensitivities,coil_axis=-3) 
      scalingfactor=tf.cast(tf.reduce_max(tf.abs(image)),image.dtype)
      image=image/scalingfactor
      uskspace=uskspace/scalingfactor

      # Convert back to image space to get zero-filled image.
      zfill = tfmri.signal.ifft(uskspace, axes=[-2, -1], norm='ortho', shift=True)

      # Combine coils and normalize zero-filled image.
      zfill=tfmri.coils.combine_coils(zfill , maps=sensitivities, coil_axis=-3)
      
      if normalize_input:
        zfill = tfmri.scale_by_min_max(zfill)  # range [0, 1]

      # Reshape into correct format N(xC)xTxHxW(x2)
      uskspace = tf.transpose(uskspace, perm= (1,0,2,3))
      sensitivities = tf.transpose(sensitivities, perm= (1,0,2,3))
      zfill = tf.stack((tf.math.real(zfill),tf.math.imag(zfill)), axis=-1)
      uskspace = tf.stack((tf.math.real(uskspace),tf.math.imag(uskspace)), axis=-1)
      sensitivities= tf.stack((tf.math.real(sensitivities),tf.math.imag(sensitivities)), axis=-1)
      image = tf.stack((tf.math.real(image),tf.math.imag(image)), axis=-1)
      mask=tf.transpose(mask, perm= (1,0,2,3))
      
    inputs2={"u_t": tf.cast(zfill,tf.float32),
            "f": tf.cast(uskspace,tf.float32),
            "coil_sens": tf.cast(sensitivities,tf.float32),
            "sampling_mask":  tf.cast(mask,tf.float32)}
    
    targets=tf.cast(image,tf.float32)
    dataset_list = [
                      tf.where(tf.math.is_nan(inputs2["u_t"]), 0., inputs2["u_t"]),
                      tf.where(tf.math.is_nan(inputs2["f"]), 0., inputs2["f"]),
                      tf.where(tf.math.is_nan(inputs2["coil_sens"]), 0., inputs2["coil_sens"]),
                      tf.where(tf.math.is_nan(inputs2["sampling_mask"]), 0., inputs2["sampling_mask"]),
                      tf.where(tf.math.is_nan(targets), 0., targets),
                    ]

    return dataset_list  
  
  return _preprocessing_fn

## Support Function
def make_fs_rtcine_image(kspace, image_shape, roll=0,phases=20,  keep_external_signal=False):
    """Returns a fully sampled image from k-space."""
    kspace=tf.ensure_shape(kspace,(None,) * 4)
    # Crop to fixed size and normalize pixel intensities (in image space).
    image = tfmri.signal.ifft(kspace, axes=[-2, -1], norm='ortho', shift=True)
    if not keep_external_signal:
      image = tfmri.resize_with_crop_or_pad(image, image_shape)

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
    image = tfmri.resize_with_crop_or_pad(image, image_shape)
    return image

