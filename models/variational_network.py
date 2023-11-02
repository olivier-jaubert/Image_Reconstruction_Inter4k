import tensorflow as tf
import tensorflow_mri as tfmri

class VarNet(tf.keras.Model):
	def __init__(self, rank=2,num_recon_blocks=10, kernel_sizes=[3], out_channels=[2], unet_layer_sizes=[32,64],**kwargs):
		"""VarNet model with UNet regulariser.

		Args:
			num_recon_blocks (`int`, optional): 
			Number of VarNet unrolling layers to use.

			kernel_sizes (`tuple`, optional): 
			The UNet kernel sizes - if passed as an integer, the kernel sizes remain fixed for each VarNet layer, otherwise each VarNet has a different UNet kernel size. If `len(kernel_sizes)` < `num_recon_blocks`, defaults to a fixed kernel size of 3.
			
			out_channels (`tuple`, optional): 
			The UNet output channel sizes - if passed as an integer, the channels remain fixed for each VarNet layer, otherwise each VarNet has a different UNet output channel size. If `len(out_channels)` < `num_recon_blocks`, defaults to a fixed output channel size of 2.

			unet_layer_sizes (`tuple`, optional): 
			The UNet sizes for each depth - defaults to a shallow UNet (2D) with 2 downsampling layers at [32,64].

		"""

		super().__init__()

		if len(kernel_sizes) < num_recon_blocks or len(out_channels) < num_recon_blocks:
			self.kernel_sizes = [3 for i in range(num_recon_blocks)]
			self.out_channels = [2 for i in range(num_recon_blocks)]
		else:
			self.kernel_sizes = kernel_sizes
			self.out_channels = out_channels
		
		self.num_recon_blocks = num_recon_blocks
		self.unet_layer_sizes = unet_layer_sizes
		self.rank = rank

		self.recon_blocks = [VNReconCellUNet(output_channels=self.out_channels[i], kernel_size=self.kernel_sizes[i], rank=self.rank,trainable=True, name=f"vn_recon_cell_unet_{i}", unet_layer_sizes=self.unet_layer_sizes,**kwargs) for i in range(self.num_recon_blocks-1)]

		# Add the last layer separately to set the `last_layer` flag to get
		# correct output from model
		self.recon_blocks.append(VNReconCellUNet(output_channels=self.out_channels[-1], kernel_size=self.kernel_sizes[-1], rank=self.rank, trainable=True, name=f"vn_recon_cell_unet_{num_recon_blocks-1}", unet_layer_sizes=self.unet_layer_sizes, last_layer=True,**kwargs))
	
	def call(self, inputs):
		x = inputs
		for layer in self.recon_blocks:
			x = layer(x)
		return x

	def train_step(self, data):
		# Unpack the data from each dataset in the list
		u_t, f, coil_sens, mask, target = data
		with tf.GradientTape() as tape:
			# Run the forward pass of the layer.
			# The operations that the layer applies
			# to its inputs are going to be recorded
			# on the GradientTape.
			logits = self([u_t, f, coil_sens, mask], training=True)  # Logits for this minibatch
			loss = self.compute_loss(y=target, y_pred=logits)

		# Use the gradient tape to automatically retrieve
		# the gradients of the trainable variables with respect to the loss.
		trainable_vars = self.trainable_variables
		grads = tape.gradient(loss, trainable_vars)
		
    	# # Run one step of gradient descent by updating
		# # the value of the variables to minimize the loss.
		self.optimizer.apply_gradients(zip(grads, trainable_vars))
		
		for metric in self.metrics:
			if metric.name == "loss":
				metric.update_state(loss)
			else:
				metric.update_state(target, logits)
		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}
	
	def test_step(self, data):
		# Unpack the data from each dataset in the list
		u_t, f, coil_sens, mask, target = data
		with tf.GradientTape() as tape:
			logits = self([u_t, f, coil_sens, mask], training=False)  # Logits for this minibatch
			loss = self.compute_loss(y=target, y_pred=logits)

		for metric in self.metrics:
			if metric.name == "loss":
				metric.update_state(loss)
			else:
				metric.update_state(target, logits)
		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}

class VNReconCellUNet(tf.keras.layers.Layer):
	""" Custom convolutional filter kernel """

	def __init__(self, output_channels, kernel_size,rank=2, unet_layer_sizes=[32,64],last_layer=False,lbd_trainable=True,lbd_init=1.0, newversion=False, **kwargs):
		super().__init__()
		self.output_channels = output_channels
		self.kernel_size = kernel_size
		#self.third_dim = 2
		#self.lamb = tf.Variable(0.5, dtype=tf.float32, trainable=True, name="ds_lambda")
		self.lamb = tf.Variable(lbd_init, dtype=tf.float32, trainable=lbd_trainable, name="ds_lambda")
		self.unet_layer_sizes = unet_layer_sizes
		self.is_last_layer = last_layer
		self.kwargs=kwargs

		#OJ
		self.newversion=newversion
		if newversion:
			self.lamb = tf.Variable(0.1, dtype=tf.float32, trainable=lbd_trainable, 
							constraint=lambda t: tf.clip_by_value(t, 0, 10**38),name="ds_lambda")# Cannot be negative
		self.rank=rank
		

	def build(self, inputs):
		in_shape = inputs[0] # shape: (None, 192, 192, 2)
		if self.rank==2:
			self.cnn_module = tfmri.models.UNet2D(self.unet_layer_sizes, self.kernel_size, out_channels=self.output_channels,**self.kwargs)
		else:
			self.cnn_module = tfmri.models.UNet3D(self.unet_layer_sizes, self.kernel_size, out_channels=self.output_channels,**self.kwargs)
		self.cnn_module.build(in_shape)

	def mri_forward_operator(self, u, coil_sens, sampling_mask):
		"""
		Forward pass with kspace
		
		Parameters:
		----------
		u: tensor NxTxHxWx2
			complex input image
		coil_sens: tensor NxCxHxWx2
			coil sensitivity map
		sampling_mask: tensor NxTxHxW
			sampling mask to undersample kspace

		Returns:
		-----------
		kspace of u with applied coil sensitivity and sampling mask
		"""

		# Expand image to get Nx1xHxWx2 to allow coil sensitivities to be applied to 
		# the image via broadcasting
		u = tf.expand_dims(u, axis=1)
		u=complexify_two_channel(u)
		coil_sens=complexify_two_channel(coil_sens)
		coil_imgs = u * coil_sens
		#coil_imgs = complexify_two_channel(coil_imgs)

		# Perform FFT on the HxW dimensions (need to make a complex tensor (1 channel))
		if self.rank==3:
			Fu = tfmri.signal.fft(coil_imgs, axes=[-3,-2,-1], shift=True,norm='ortho')
		else:
			Fu = tfmri.signal.fft(coil_imgs, axes=[-2,-1], shift=True,norm='ortho')
		Fu = realify_complex(Fu)

		#mask = tf.expand_dims(sampling_mask, 1) # Nx1xTxHxW
		mask = tf.expand_dims(sampling_mask, -1) # Nx1xTxHxWx1

		# Repeat the undersampling mask over the expanded dimension (the real / complex 
		# dimension) twice (once for real, once for complex)
		mask = tf.repeat(mask, repeats=2, axis=-1) # Nx1xTxHxWx2
		#print(mask.shape,Fu.shape)
		# Apply undersampling mask to coil images in kspace
		kspace = mask*Fu # NxCxHxWx2
		
		return kspace

	def mri_adjoint_operator(self, f, coil_sens):
		"""
		Adjoint operation that convert kspace to coil-combined under-sampled image
		by using coil_sens and sampling mask
		
		Parameters:
		----------
		f: tensor NxCxHxWx2
			multi channel undersampled kspace
		coil_sens: tensor NxCxHxWx2
			coil sensitivity map

		Returns:
		-----------
		Undersampled, coil-combined image
		"""
		
		f = complexify_two_channel(f)

		
		if self.rank==3:
			Finv = tfmri.signal.ifft(f, axes=[-3,-2,-1], shift=True,norm='ortho') # NxCxTxHxW (complex64)
		else:
			Finv = tfmri.signal.ifft(f, axes=[-2,-1], shift=True,norm='ortho') # NxCxTxHxW (complex64)
		# multiply coil images with sensitivities and sum up over channels
		coil_sens = complexify_two_channel(coil_sens)
		
		# Weighted sum of squares coil combination (adaptive)
		img = tfmri.combine_coils(Finv, maps=coil_sens, coil_axis=1)
		img = realify_complex(img)
		return img #NxTxHxWx2
	
	def call(self, inputs):
		"""
		Forward pass for variational layer
		
		Parameters:
		----------
		inputs: list containing current image (u_t), 
		kspace (f), coil sensitivities (c) and 
		undersampling mask (m)

		Returns:
		-----------
		Output list consisting of new image, new ksapce, coil sensitivieies and sampling mask
		"""

		u_t_1 = inputs[0] #NxTxHxWx2
		f = inputs[1] #NxCxTxHxWx2
		c = inputs[2] #NxCx1xHxWx2
		m = inputs[3] #NxTxHxW

		# Apply CNN regulariser
		Ru = self.cnn_module(u_t_1) #NxTxHxWx2

		Au = self.mri_forward_operator(u_t_1, c, m)

		At_Au_f = self.mri_adjoint_operator(Au - f, c)
		
		USE_FIXED_DC_LAMBDA = False
		if USE_FIXED_DC_LAMBDA:
			Du = At_Au_f * 0.2
			u_t = u_t_1 - Du - self.lamb*Ru
		else:
			Du = At_Au_f * self.lamb
			if self.newversion:
				u_t = u_t_1 - Du + Ru
			else:
				u_t = u_t_1 - Du - Ru
			
		
		output = [u_t,f,c,m]
		
		if self.is_last_layer:
			return output[0]
		else:
			return output #NxCxHxWx2
	
# Utils for variational Network	
def complexify_two_channel(x):
	# Returns complex tensor from two channel input tensor
	# of the form NxHxWx2
	return tf.complex(x[...,0], x[...,1])

def realify_complex(x):
	# Returns two channel real tensor from complex input tensor
	# of the form NxHxW
	return tf.stack((tf.math.real(x), tf.math.imag(x)), axis=-1)

# Define custom loss (SSIM on magnitude)
def custom_loss_ssim(y_true, y_pred, use_mean=False, norm=False):
    # Custom SSIM loss on magnitude images (pixel range [0,1] now, not [-1,1])
    # returns a tensor of losses for each image in the batch 
    true = tf.expand_dims(tf.abs(tf.complex(y_true[...,0], y_true[...,1])), axis = -1)
    pred = tf.expand_dims(tf.abs(tf.complex(y_pred[...,0], y_pred[...,1])), axis = -1)
    if norm:
        true /= tf.reduce_max(true)
        pred /= tf.reduce_max(pred)
        
    if use_mean:
        return 1 - tf.reduce_mean(tf.image.ssim(true, pred, 1.0))
    else:
        return 1 - tf.image.ssim(true, pred, 1.0)

# Define metric (SSIM on magnitude)
def custom_metric_ssim(y_true, y_pred, use_mean=False, norm=False):
	# Custom SSIM loss on magnitude images (pixel range [0,1] now, not [-1,1])
	# returns a tensor of losses for each image in the batch 
	true = tf.expand_dims(tf.abs(tf.complex(y_true[...,0], y_true[...,1])), axis = -1)
	pred = tf.expand_dims(tf.abs(tf.complex(y_pred[...,0], y_pred[...,1])), axis = -1)

	if norm:
		true /= tf.reduce_max(true)
		pred /= tf.reduce_max(pred)
		
	if use_mean:
		return tf.reduce_mean(tf.image.ssim(true, pred, 1.0))
	else:
		return tf.image.ssim(true, pred, 1.0)

# Define custom loss (MSE on magnitude)
def custom_loss_mse(y_true, y_pred, norm=False):
	true = tf.expand_dims(tf.abs(tf.complex(y_true[...,0], y_true[...,1])), axis = -1)
	pred = tf.expand_dims(tf.abs(tf.complex(y_pred[...,0], y_pred[...,1])), axis = -1)	
	if norm:
		true /= tf.reduce_max(true)
		pred /= tf.reduce_max(pred)
	
	loss = tf.square(true-pred)
	return tf.reduce_mean(loss)
