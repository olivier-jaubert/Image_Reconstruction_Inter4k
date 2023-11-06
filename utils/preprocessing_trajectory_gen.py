import tensorflow as tf
import tensorflow_mri as tfmri

def config_optimized_traj():
    config={'flow': 0,
            'base_resolution': 240,
            'field_of_view': 400,
            'phases': 12,
            'ordering': 'linear',   
            'max_tempres': 55,
            'min_max_arm_time': [0.88,1.67],
            'vd_spiral_arms': 16,
            'vd_inner_cutoff': 0.15,
            'pre_vd_outer_cutoff': 0.41288,
            'vd_outer_density': 0.07,
            'vd_type': 'hanning',
            'max_grad_ampl': 22.0,
            'min_rise_time': 10.0,
            'dwell_time': 1.4,
            'gradient_delay': 0.56,
            'readoutOS': 2.0,
            'deadtime': 2.0,
            'reverse': True,
    }
    return config

def config_radial_traj():
    config={'radial_spokes':13,
        'base_resolution': 256,
            'field_of_view': 400,
            'phases': 32,
            'ordering': 'tiny_half',  
            'reverse': True,
    }
    return config

def gen_spiral_traj(
    base_resolution=256,
	field_of_view=400,phases=1,
	ordering='golden',tiny_number=7,
	max_tempres=50,min_max_arm_time=[3,8],
	vd_spiral_arms=26,vd_inner_cutoff=0.2,
	pre_vd_outer_cutoff=0.5,vd_outer_density=0.25,
	vd_type='linear',
    max_grad_ampl=24.0,
    min_rise_time=8.125,
    dwell_time=1.4,
    gradient_delay=0.56,readoutOS=2,deadtime=1.9,larmor_const=42.577478518,flow=0):
        """Returns a spiral trajectory.
        Computes 1 arm, if conditions min_max_arm_time respected:
            computes the full trajectory -> (nphases, narms, nsamples, 2)
            and density compensation weights -> (nphases, narms, nsamples)
            and time_for_an_arm -> float
        """
        views=1
        if vd_inner_cutoff>0.9:
            vd_outer_cutoff=1
            vd_outer_density=1
        else:
            vd_outer_cutoff=float(vd_inner_cutoff+0.1+pre_vd_outer_cutoff*(1-vd_inner_cutoff-0.1))
        #Generate 1 spiral arm
        traj_var=tfmri.spiral_trajectory(
            base_resolution, vd_spiral_arms, field_of_view, 
            max_grad_ampl, min_rise_time, dwell_time,
            views=1,phases=1,ordering=ordering,
            tiny_number=tiny_number,
            vd_inner_cutoff=vd_inner_cutoff,
            vd_outer_cutoff=vd_outer_cutoff,
            vd_outer_density=vd_outer_density,
            gradient_delay=gradient_delay,vd_type=vd_type,readout_os=readoutOS,larmor_const=larmor_const)
        
        # If within the specified time for an arm allowed generates the whole trajectory otherwise returns incomplete traj

        time_for_an_arm=traj_var.shape[2]*dwell_time*10**-3
        dcwvar=tf.ones((1,1,1))

        if (time_for_an_arm<min_max_arm_time[1] and time_for_an_arm>min_max_arm_time[0]) or vd_spiral_arms==1:
            
            TR=time_for_an_arm+deadtime
            if flow:
                #For Flow more time for velocity encoding ->+3.6
                TR=time_for_an_arm+3.6
            
            #determine number of spiral arms depending on max temporal resolution and calculated TR.
            views=int(max_tempres/TR)
            #Generate trajectory
            traj_var=tfmri.spiral_trajectory(
                base_resolution, vd_spiral_arms, field_of_view, 
                max_grad_ampl, min_rise_time, dwell_time,
                views=views,phases=phases,ordering=ordering,
                tiny_number=tiny_number,
                vd_inner_cutoff=vd_inner_cutoff,
                vd_outer_cutoff=vd_outer_cutoff,
                vd_outer_density=vd_outer_density,
                gradient_delay=gradient_delay,vd_type=vd_type,readout_os=readoutOS,larmor_const=larmor_const)
            
            #Density Estimation
            traj_var_full=tfmri.spiral_trajectory(
                base_resolution, vd_spiral_arms, field_of_view, 
                max_grad_ampl, min_rise_time, dwell_time,
                ordering=ordering,tiny_number=tiny_number,
                views=vd_spiral_arms,
                vd_inner_cutoff=vd_inner_cutoff,
                vd_outer_cutoff=vd_outer_cutoff,
                vd_outer_density=vd_outer_density,
                gradient_delay=gradient_delay,vd_type=vd_type,readout_os=readoutOS,larmor_const=larmor_const)
            #truncate center point (gradient delay)
            traj_var=traj_var[:,:,1:,:]
            traj_var_full=traj_var_full[:,1:,:]
            #Reshaping trajectory:
            var_shape=traj_var.shape
            var_shape_full=traj_var_full.shape
            traj_var_full=tf.reshape(traj_var_full, (1,var_shape_full[0]* var_shape_full[1], var_shape_full[2]))
            dens=tfmri.estimate_density(traj_var_full[0:1,...], [base_resolution,base_resolution],method='pipe',max_iter=30)
            dcwvarfull=tf.math.divide_no_nan(1.0,dens)
            traj_var_full=tf.reshape(traj_var_full, var_shape_full)
            dcwvarfull=tf.reshape(dcwvarfull,var_shape_full[0:-1])
            dcwvarfull=tf.tile(tf.expand_dims(dcwvarfull,0),(var_shape[0],)+(1,1,))
            
            if views>vd_spiral_arms:
                dcwvarfull=tf.tile(dcwvarfull,(1,views,1,))

            dcwvar=dcwvarfull[:,0:views,...]
        
        return traj_var, dcwvar, time_for_an_arm

def create_traj_fn( radial_spokes=0,
                    flow=0,
                    base_resolution=256,
                    field_of_view=400,
                    phases=1,
                    ordering='golden',
                    tiny_number=7,
                    max_tempres=50,
                    min_max_arm_time=[3,8],
                    vd_spiral_arms=26,
                    vd_inner_cutoff=0.2,
                    pre_vd_outer_cutoff=0.5,
                    vd_outer_density=0.25,
                    vd_type='linear',
                    reverse= False,
                    max_grad_ampl=24.0,
        	    min_rise_time=8.125,
        	    dwell_time=1.4,
        	    gradient_delay=0.56,
        	    readoutOS=2,deadtime=1.9,larmor_const=42.577478518):

    """Returns a preprocessing function generating spiral trajectory.
        For HyperBand Optimization, 
        generates spiral trajectory and adjusts depending on wether it fits the min_max_arm_time allowed
    """
    def _create_traj(inputs):
        
        max_guesses=50
        counter=0
        time_for_an_arm=0
        vd_spiral_arms0=vd_spiral_arms
        
        if radial_spokes>0:
            #Generate Radial trajectory
            traj_params = {
            'base_resolution': base_resolution,
            'views': radial_spokes,
            'phases': phases,
            'ordering': ordering,
            }
            traj = tfmri.sampling.radial_trajectory(**traj_params)
            dens = tfmri.sampling.estimate_radial_density(traj)
            dcw = tf.math.divide_no_nan(1.0,dens)
            time_for_an_arm=(min_max_arm_time[1]+min_max_arm_time[0])/2
        else:
            #Try while conditions not respected to generate trajectory with input parameters (until max guesses)
            #Generate Spiral trajectory
            while not (time_for_an_arm<min_max_arm_time[1] and time_for_an_arm>min_max_arm_time[0]) and counter<max_guesses and vd_spiral_arms0>0:
                
                traj,dcw,time_for_an_arm=gen_spiral_traj(flow=flow,base_resolution=base_resolution,field_of_view=field_of_view,phases=phases,max_tempres=max_tempres,ordering=ordering,tiny_number=tiny_number,min_max_arm_time=min_max_arm_time,
                                            vd_spiral_arms=vd_spiral_arms0,vd_inner_cutoff=vd_inner_cutoff,
                                            pre_vd_outer_cutoff=pre_vd_outer_cutoff,
                                            vd_outer_density=vd_outer_density,vd_type=vd_type,
                                            max_grad_ampl=max_grad_ampl,
                                            min_rise_time=min_rise_time,
                                            dwell_time=dwell_time,
                                            gradient_delay=gradient_delay,
                                            readoutOS=readoutOS,
                                            deadtime=deadtime,larmor_const=larmor_const)

                counter+=1
                if time_for_an_arm<min_max_arm_time[0]:
                    vd_spiral_arms0-=1
                elif time_for_an_arm>min_max_arm_time[1]:
                    vd_spiral_arms0+=1

        if reverse== True:
            traj= tf.reverse(traj, [-1])        
        trajectory=dict()
        trajectory['traj']=traj
        trajectory['dcw']=dcw
        #Attach trajectory to ground truth kspace.
        dataset_traj=tf.data.Dataset.from_tensors(trajectory).repeat(-1)
        ds= tf.data.Dataset.zip((inputs,dataset_traj))
        ds = ds.map(lambda image, traj: {'image': image, 'traj': traj})
        return ds
        
    return _create_traj
  
