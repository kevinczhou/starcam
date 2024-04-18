import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm.notebook import tqdm
import scipy.signal
from mcam3d import mcam3d
from pathlib import Path
from mcam_loading_scripts import filepath_key
import xarray as xr
import h5py


class zstitch(mcam3d):
    # zstitch in principle is a straightforward extension of mcam3d to obtain depth from stitching z-stacks rather than
    # infer parallax
    def __init__(self, z_step_mm, nominal_z_slices=None, truncate=3, sigma=8, z_stage_up=True,
                 camera_dims=(9, 6), sigma_for_registration=None, *args, **kwargs):
        # nominal_z_slices: in the flat calibration reference dataset, which z slices were chosen for stitching?
        # z_stage_up: specifies whether the stage is moving upwards
        # z_step_mm: step size in z in mm
        # sigma_for_registration: used if use_hpf_for_MSE_loss is set to True
        super().__init__(*args, **kwargs)
        self.z_step_mm = z_step_mm
        self.nominal_z_slices = nominal_z_slices
        self.unet_scale = 1
        self.z_stage_up = z_stage_up
        self.camera_dims = camera_dims

        # truncate and sigma are variables for _create_filters:
        self.truncate = truncate  # after how many sigmas do you truncate the gaussian?
        self.sigma = sigma  # for gaussian filter
        self.gaussian_kernel = self._create_filters(sigma)  # create the filters
        # derivative kernels:
        self.diff0_kernel = np.array([[1, -1]])
        self.diff1_kernel = self.diff0_kernel.T

        if sigma_for_registration is not None:
            self.sigma_for_registration = sigma_for_registration
            self.gaussian_kernel2 = self._create_filters(sigma_for_registration)
            self.gaussian_kernel2_sqrt2 = self._create_filters(sigma_for_registration * np.sqrt(2))

        self.weighted_sharpness_loss = False  # looking at the z profile of the sharpness maps
        self.weighted_sharpness_thresholds = [1, 1.5]  # for excluding positions where the sharpness metric isn't high;
        # generally, the second one should be larger

    def _create_filters(self, sigma):
        # filters for computing sharpness metric for z-stacks

        # gaussian blur kernel:
        w = 2 * int(self.truncate * sigma + 0.5) + 1
        x = np.arange(-w // 2, w // 2) + 1
        x, y = np.meshgrid(x, x)
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / 2 / sigma ** 2)
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

        return gaussian_kernel

    def generate_patched_dataset_from_disk(self, filepath, num_patches, patch_size, nominal_z_slices_global,
                                           patch_recon_size=None,
                                           sample_margin=None, prefetch=5, cam_slice0=(0, 9), cam_slice1=(0, 6),
                                           inclusive_patch_selection=False,
                                           ):
        # This is a copy of generate_patched_dataset, modified to load patches from storage rather than from RAM.
        # As such, fracture_big_tensors is no longer necessary.
        # prefetch should be a larger value to average out load times for uneven number of zstacks per batch.
        # Otherwise, most things are similar to generate_patched_dataset.
        # cam_slice0/1 specify which cameras to sample from; they are length-2 lists or tuples specifying start and end
        # of slices.
        # filepath specifies where the dataset is.
        # inclusive_patch_selection: when retrieving from the visitation log, check intersection with any point within
        # r:r+patch_size, c:c+patch_size (i.e., old behavior); otherwise, only check r,c.
        # nominal_z_slices_global: for all 54*64 lateral positions; these nominal z slices include the z_step_ratio!

        self.nominal_z_slices_global = nominal_z_slices_global
        self.nominal_z_slices_global_unflattened = tf.constant(nominal_z_slices_global.reshape(self.array_dims))
        # ^needs to be a tf tensor for TensorArray.write;

        # get names of the xy scan folders (e.g., 8x8 scan = 64 folders), each containing a .nc file of up to 54 camera
        # snapshots:
        directories = list(d for d in Path(filepath).iterdir() if d.is_dir() and d.name[0] == 'y')
        directories = sorted(directories, key=filepath_key)

        y_name_initial = '_'.join(directories[0].name.split('_')[:2])
        x_name_initial = '_'.join(directories[0].name.split('_')[2:])
        y_steps = sum(d.name.startswith(y_name_initial) for d in directories)
        x_steps = sum(d.name.endswith(x_name_initial) for d in directories)
        directories = np.asarray(directories, dtype=object).reshape(y_steps, x_steps)
        self.directories = directories  # array (e.g., 8x8) of directories containing the .nc files
        # get the z_stack in each directory:
        self.directories_list = list()  # list version, the one that will be used by tf to select the filename
        self.open_dataset_list = list()  # pre-open datasets
        self.h5file_list = list()  # the output of h5py.File()
        for i in range(self.directories.shape[0]):
            for j in np.arange(self.directories.shape[1])[::-1]:  # this dim reversed to make scan direction consistent
                # convert from posixpath to string so that tf supports the dtype:
                filename = str(next(self.directories[i, j].glob('z_stack*')))
                self.directories[i, j] = filename
                self.directories_list.append(filename)

                h5file = h5py.File(filename, 'r')
                if 'mcam_data' in h5file:
                    dataset = h5file.get('mcam_data')
                else:
                    dataset = h5file.get('images')

                self.h5file_list.append(h5file)
                self.open_dataset_list.append(dataset)
        # preallocate array for h5py.read_direct (faster to keep the singleton dimensions):
        self.preallocated = np.full((self.num_channels, 1, 1, patch_size, patch_size), fill_value=0, dtype='uint8')

        # create rc_warper -- an instance of the zstitch class itself -- to generate rc_warp one image at a time (the
        # outer instance of zstitch is used for creating the visitation logs, which operate on many (e.g., 64) images;
        self.rc_warper = zstitch(stack=np.zeros((1,) + self.stack.shape[1:3] + (1,)),
                                 ul_coords=np.zeros((1, 2)),  # (<^) these don't matter
                                 recon_shape=self.recon_shape, ul_offset=self.ul_offset, z_step_mm=self.z_step_mm,
                                 scale=.01, batch_size=None, momentum=None, report_error_map=False)
        self.rc_warper.create_variables(deformation_model='camera_parameters',
                                        learning_rates={'camera_focal_length': -1e-3, 'camera_height': -1e-3,
                                                        'ground_surface_normal': -1e-3, 'camera_in_plane_angle': -1e-3,
                                                        'rc': -.1, 'gain': -1e-3, 'bias': -1e-3, 'illum_flat': -1e-3,
                                                        'radial_camera_distortion': -1e-3},
                                        variable_initial_values=None,  # will be adjusted later
                                        remove_global_transform=False, antialiasing_filter=False)
        self.rc_warper_camera_params_dict = {v.name[:-2]: v for v in self.rc_warper.train_var_list}  # to access
        # variables easily

        # camera parameters shared by all cameras can be assigned now:
        self.rc_warper_camera_params_dict['radial_camera_distortion'].assign(
            self.variable_initial_values_global['radial_camera_distortion'][None, None])
        # this is what will be passed to warp_camera_parameters later in the tf dataset:
        self.rc_warper.rc_base = np.transpose(self.rc_warper.rc_base, (0, 2, 1, 3))

        self.num_patches = num_patches
        self.patch_size = patch_size
        assert patch_size % 2 == 0  # we'll be dividing this by 2 in _gather_image_patches_from_disk
        self._validate_patch_size()
        if patch_recon_size is None:
            self.patch_recon_size = 3 * self.patch_size
        else:
            self.patch_recon_size = patch_recon_size

        # run the network once so that we can access network.trainable_variables
        if self.use_postpended_channels_for_stitching:
            num_channels = self.num_channels - self.num_channels_rgb  # self.num_channels_rgb channels postpended, which
            # are for registration, not CNN input
        else:
            num_channels = self.num_channels
        out = self.network(tf.zeros([1, self.patch_size, self.patch_size, num_channels], dtype=self.tf_dtype))
        self.output_patch_size = out.numpy().shape[1]  # might not be same as self.patch_size if not using padded convs
        print('Output patch size: ' + str(self.output_patch_size))
        if self.patch_size != self.output_patch_size:
            print('Warning: training will work with output != input size, but inference on full images will not')

        if sample_margin is not None:
            row_low = sample_margin / 2 * self.recon_shape_base[0]
            row_high = (1 - sample_margin / 2) * self.recon_shape_base[0] - self.patch_size - 1
            col_low = sample_margin / 2 * self.recon_shape_base[1]
            col_high = (1 - sample_margin / 2) * self.recon_shape_base[1] - self.patch_size - 1
        else:
            row_low = 0
            row_high = self.recon_shape_base[0] - self.patch_size - 1
            col_low = 0
            col_high = self.recon_shape_base[1] - self.patch_size - 1

        # tf complains if this lambda function isn't defined on a standalone line:
        # (recon_shape_base is the non-downsampled size)
        generate_rand_coord = lambda x: (tf.random.uniform((), cam_slice0[0],
                                                           cam_slice0[1], dtype=tf.int32),  # camera array dim 0
                                         tf.random.uniform((), cam_slice1[0],
                                                           cam_slice1[1], dtype=tf.int32),  # camera array dim 1
                                         tf.random.uniform((), row_low, row_high),  # coordinates of visitation log
                                         tf.random.uniform((), col_low, col_high))
        # need to wrap this with a tf.py_function, since it uses non-tf functions:
        gather_image_patches_from_disk = lambda cam0, cam1, r, c: self._gather_image_patches_from_disk(
            cam0, cam1, r, c, inclusive_patch_selection)
        if inclusive_patch_selection:
            print('WARNING: inclusive_patch_selection behavior has changed and not been tested (01/09/2022)!')

        dataset = (tf.data.Dataset.range(1)  # dummy dataset
                   .map(generate_rand_coord)  # generate one random coordinate (camera and pixel)
                   .map(gather_image_patches_from_disk, num_parallel_calls=-1, deterministic=False)  # don't think num_parallel or deterministic helps
                   .repeat(None)  # generate infinite number of patches
                   .apply(tf.data.experimental.dense_to_ragged_batch(batch_size=self.num_patches))  # different number
                   # of image patches per reconstruction patch
                   .prefetch(prefetch)
                   )  # ragged function seems to only work in tf2.3 (2.2 fails)

        self.d0, self.d1 = np.indices([self.patch_size, self.patch_size])  # for help with gather_nd later

        return dataset

    def _gather_image_patches_from_disk(self, cam0, cam1, r, c, inclusive_patch_selection):
        # a copy of _gather_image_patches, but modified to accompany generate_patched_dataset_from_disk
        # cam0 and cam1 specify which camera to slice using r and c
        # row and col slicing from the .nc file need to be even to enable consistent debayering

        r = tf.cast(r * self.visitation_log_scale, dtype=tf.int32)
        c = tf.cast(c * self.visitation_log_scale, dtype=tf.int32)

        # get the visitation log for cam0, cam1:
        visitation_log_r = self.visitation_logs_r_for_all_cameras[cam0, cam1]
        visitation_log_c = self.visitation_logs_c_for_all_cameras[cam0, cam1]

        # retrieve records from visitation log:
        # remember that the visitation log has -1 for unvisited pixels!
        patch_size_scaled = tf.cast(self.patch_size * self.visitation_log_scale, dtype=tf.int32)
        if inclusive_patch_selection:
            retrieved_record_r = visitation_log_r[r:r + patch_size_scaled, c:c + patch_size_scaled]
            retrieved_record_c = visitation_log_c[r:r + patch_size_scaled, c:c + patch_size_scaled]
            # shapes: (patch_size, patch_size, num_images)
        else:
            r += patch_size_scaled // 2  # center the coordinate; sampling is based on upper left position
            c += patch_size_scaled // 2
            retrieved_record_r = visitation_log_r[r:r + 1, c:c + 1]
            retrieved_record_c = visitation_log_c[r:r + 1, c:c + 1]

        max_r = tf.reduce_max(retrieved_record_r, axis=(0, 1))  # shape: num_images
        max_c = tf.reduce_max(retrieved_record_c, axis=(0, 1))

        # If max_r/c is less than the max dim of images and greater than patch_size, then crop image from max-patch_size
        # to max; if max is less than patch_size, then crop image from 0 to patch_size; if max is greater than
        # image size, then crop from image_size-patch size to image_size.

        # first, filter images by those which visited the current patch (if unvisited, max_r will be -1)
        inds_images_to_use = tf.cast(tf.where(max_r >= 0)[:, 0], tf.int32)  # length < num_images

        # use a for-loop here, at least to avoid doing a messy tf.gather operation
        im_patches = tf.TensorArray(tf.uint8, size=len(inds_images_to_use),  # use in lieu of list
                                    element_shape=(self.patch_size, self.patch_size, self.num_channels))
        rc_warp_patches = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                         element_shape=(self.patch_size, self.patch_size, 2))
        cam_to_vanish_batch = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                             element_shape=())
        vanish_point_batch = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                            element_shape=(2,))
        nominal_z_slices_batch = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                                element_shape=())
        inds_images_to_use_ = tf.TensorArray(tf.int32, size=len(inds_images_to_use),
                                             element_shape=())  # this may seem useless, but for some reason tf doesn't
        # combine inds_images_to_use into a ragged batch (at least in tf2.3); tf.zeros_like(inds_images_to_use) also
        # fails, but tf.zeros(len(inds_images_to_use)) succeeds

        for i in tf.range(len(inds_images_to_use)):  # can't use enumerate, or else tf might interpret as a python loop
            ind = inds_images_to_use[i]
            max_r_ind = max_r[ind]  # one number
            max_c_ind = max_c[ind]  # one number

            # three cases:
            # 1) max_r_ind < self.patch_size / 2
            # 2) max_r_ind >= self.stack.shape[1] - self.patch_size / 2
            # 3) self.patch_size <= max_r_ind < self.stack.shape[1] (treated as default below)
            # (and the same one for c)
            # note that patch_size is even
            r_start, r_end = tf.case([(tf.less(max_r_ind, self.patch_size // 2), lambda: (0, self.patch_size)),
                                      (tf.greater_equal(max_r_ind, self.stack.shape[1] - self.patch_size // 2),
                                       lambda: (self.stack.shape[1] - self.patch_size, self.stack.shape[1])),
                                      ], default=lambda: (max_r_ind - self.patch_size // 2,
                                                          max_r_ind + self.patch_size // 2))

            c_start, c_end = tf.case([(tf.less(max_c_ind, self.patch_size // 2), lambda: (0, self.patch_size)),
                                      (tf.greater_equal(max_c_ind, self.stack.shape[2] - self.patch_size // 2),
                                       lambda: (self.stack.shape[2] - self.patch_size, self.stack.shape[2])),
                                      ], default=lambda: (max_c_ind - self.patch_size // 2,
                                                          max_c_ind + self.patch_size // 2))

            # ensure r_start and c_start are even:
            r_is_even = tf.math.floormod(r_start, 2) == 0
            r_start, r_end = tf.cond(r_is_even, lambda: (r_start, r_end), lambda: (r_start - 1, r_end - 1))
            c_is_even = tf.math.floormod(c_start, 2) == 0
            c_start, c_end = tf.cond(c_is_even, lambda: (c_start, c_end), lambda: (c_start - 1, c_end - 1))

            # get patch zstack:
            ind_2d = tf.py_function(lambda x: np.unravel_index(x, self.xy_scans), inp=[ind], Tout=[tf.int32, tf.int32])
            # ^unflatten index; output of this is a list
            patch = tf.py_function(self._load_one_zstack_patch, inp=[ind, cam0, cam1, r_start, c_start],
                                   Tout=tf.uint8)

            # get rc_warp:
            # (make sure you've run generate_visitation_log_for_all_cameras first, which loads camera parameters)
            # first, replace the camera parameters:
            r_im = cam0 * self.xy_scans[0] + ind_2d[0]  # xy_scans, e.g., (8,8)
            c_im = cam1 * self.xy_scans[1] + ind_2d[1]
            for var_name in self.camera_parameter_names:  # remember this list is of params that differ per image
                # this is a tricky part: get coordinate of one of the 54*64 images using camera coordinates and
                # within that camera, an xy-scan position
                params = tf.py_function(lambda x, y: self.variable_initial_values_global[var_name][x, y],
                                        inp=[r_im, c_im], Tout=tf.float32)
                params = params[None]  # params in above line had its first two dims (camera dims) removed, so add back
                # one, which represents the flattened camera/image dims, but there's only 1
                self.rc_warper_camera_params_dict[var_name].assign(params)  # modify the parameter
            # next, generate rc_warp_dense:
            rc_base = tf.slice(self.rc_warper.rc_base, begin=[0, r_start, c_start, 0],
                               size=[-1, self.patch_size, self.patch_size, -1])
            num_images = 1  # always 1; we're in a for-loop
            rc_warp_dense = self.rc_warper._warp_camera_parameters(tf.cast(tf.reshape(rc_base, [num_images, -1, 2]),
                                                                           dtype=self.tf_dtype),
                                                                   use_radial_deformation=False)
            rc_warp_dense = tf.reshape(rc_warp_dense, (num_images, self.patch_size, self.patch_size, 2))
            rc_warp_dense /= self.rc_warper.scale  # remove the scale

            rc_warp_patches = rc_warp_patches.write(i, rc_warp_dense[0])  # remove first dim of rc_warp_dense
            im_patches = im_patches.write(i, patch)
            # tf complains if I use self.stack (numpy version) rather than self.stack_tf (tf version)^

            # these variables are small; no need to give the fracture treatment in mcam3d.py
            inds_images_to_use_ = inds_images_to_use_.write(i, ind)
            vanish_point_batch = vanish_point_batch.write(i, self.vanish_point[ind])  # not used; was used in mcam3d.py
            cam_to_vanish_batch = cam_to_vanish_batch.write(i, self.cam_to_vanish[ind])
            nominal_z_slices_batch = nominal_z_slices_batch.write(i, self.nominal_z_slices_global_unflattened[r_im, c_im])

        return (im_patches.stack(),  # shape: _ by patch_size by patch_size by 3
                rc_warp_patches.stack(),  # shape: _ by patch_size by patch_size by 2
                vanish_point_batch.stack(),  # shape: _ by 2
                cam_to_vanish_batch.stack(),  # shape: _
                nominal_z_slices_batch.stack(),  # shape: _
                inds_images_to_use_.stack(),  # shape: _
                r, c)  # also return the random coordinate

    def _load_one_zstack_patch(self, filename_ind, cam0, cam1, r, c):
        # cam0/1 are the camera coordinates
        # r and c are the starting pixels row and col coordinates
        # this function is called by _gather_image_patches_from_disk
        # this needs to be wrapped by tf.py_function, as it is being used by tf.dataset

        dataset = self.open_dataset_list[filename_ind]
        the_slice = np.s_[:, cam0:cam0 + 1, cam1:cam1 + 1, r:r + self.patch_size, c:c + self.patch_size]
        dataset.read_direct(self.preallocated, source_sel=the_slice)
        patch_z_stack = self.preallocated.squeeze()  # has two singleton dimensions for camera dimensions

        # debayer the z-stack:
        debayered = np.empty_like(patch_z_stack)
        for i, patch in enumerate(patch_z_stack):
            debayered[i] = cv2.cvtColor(np.asarray(patch), cv2.COLOR_BAYER_GB2BGR)[:, :, 1]  # only need green channel
        debayered = np.transpose(debayered, (1, 2, 0))  # move z-stack dimension to end
        return debayered

    def generate_visitation_log_for_all_cameras(self, restore_path, restrict_bounds=False,
                                                cam_slice0=(0, 9), cam_slice1=(0, 6), reuse_log=False,
                                                preferred_camera=(0, 0)):
        # If you're going to stream from disk, generate and store visitation logs for all cameras (or rather those that
        # are chosen by cam_slice0/1; can't store rc_warp_dense, unfortunately, because it's too much data, since they
        # need to be 32-bit for all 54*64 images.
        # Run this function instead of generate_visitation_log.
        # Parts of this function are adapted from zstitch_gigamosaic's load_camera_parameters function.
        # Run this on CPU.
        # restore_path: where the precalibration parameters are stored from all cameras.
        # reuse_log: if True, then repeat the first visitation log across all cameras.
        # preferred_camera: this function calls self._generate_visitation_log in a for loop, which generates
        # rc_warp every time, but doesn't use it (it's only used when not streaming data from disk), EXCEPT for
        # generating the "full" reconstruction (e.g., 8x8 scanned images); when generating this "full" reconstruction,
        # the user will pick one camera to do it on. preferred_camera chooses that camera, and will ensure that
        # self.rc_warp is for that camera. If reuse_log is True, then preferred_camera will pretty much be ignored.

        # FIRST, load the camera parameters for all cameras, to be assigned to the tf.Variables; this code was heavily
        # adapted from zstitch_gigamosaic's load_camera_parameters function:
        restored = scipy.io.loadmat(restore_path)
        # get precalibrated values:
        self.variable_initial_values_global = {key: restored[key].squeeze() for key in restored if '__' not in key}
        # get other settings:
        self.inds_keep = restored['inds_keep__'].flatten()
        # do not replace recon_shape and ul_offset from restored!
        self.array_dims = tuple(restored['camera_array_dims__'].flatten())  # cameras and lateral scans
        self.nominal_z_slices = restored['nominal_z_slices__'].flatten().astype(np.float32)[self.inds_keep]
        self.pre_downsample_factor = restored['pre_downsample_factor__'].flatten()

        assert len(self.inds_keep) == np.prod(self.array_dims)
        # doesn't handle the case when inds_keep is not the full array ...

        self.xy_scans = (self.array_dims[0] // self.camera_dims[0],
                         self.array_dims[1] // self.camera_dims[1])  # dims of xy scan (e.g., 8, 8)

        # reshape camera parameters:
        V = self.variable_initial_values_global
        V['camera_height'] = V['camera_height'].reshape(self.array_dims)
        V['ground_surface_normal'] = V['ground_surface_normal'].reshape(self.array_dims + (3,))
        V['camera_in_plane_angle'] = V['camera_in_plane_angle'].reshape(self.array_dims)
        V['rc'] = V['rc'].reshape(self.array_dims + (2,)) * self.pre_downsample_factor
        V['illum_flat'] = V['illum_flat'].reshape(self.array_dims + (6,))  # may not use this
        self.camera_parameter_names = ['camera_height', 'ground_surface_normal', 'camera_in_plane_angle',
                                       'rc']  # note: this list only contains per-camera parameters;
        # parameters that are shared among all parameters are handled separately (e.g., radial_camera_distortion).
        # I excluded illum_flat.

        self.camera_params_dict = {v.name[:-2]: v for v in self.train_var_list}  # to access variables easily
        # camera parameters shared by all cameras can be assigned now:
        self.camera_params_dict['radial_camera_distortion'].assign(V['radial_camera_distortion'][None, None])
        # note that ul_offset doesn't need to be set, because when the zstitch object was created, you should have
        # already set the right value

        # SECOND, generate the visitation logs:
        visitation_logs_r = list()
        visitation_logs_c = list()
        first_log = True  # relevant if reuse_log
        rc_warp_dense_to_remember = None
        for cam_r in tqdm(range(cam_slice0[0], cam_slice0[1])):  # e.g., 0 to 9
            for cam_c in tqdm(range(cam_slice1[0], cam_slice1[1])):  # e.g., 0 to 6
                # first, modify the camera parameters:
                for var_name in self.camera_parameter_names:  # remember this list is of params that differ per image
                    params = self.variable_initial_values_global[
                                 var_name][cam_r * self.xy_scans[0]:(cam_r+1) * self.xy_scans[0],
                                           cam_c * self.xy_scans[1]:(cam_c+1) * self.xy_scans[1]]
                    params = params.reshape(-1, *params.shape[2:])  # reflatten the two camera dims
                    self.camera_params_dict[var_name].assign(params)  # modify the parameter
                # now, we can retrieve the visitation logs:
                if first_log or not reuse_log:  # always run the first time; thereafter, only run if not reusing log
                    visitation_log_r, visitation_log_c, _, _ = self.generate_visitation_log(restrict_bounds)
                    first_log = False
                # save visitation log:
                visitation_logs_r.append(visitation_log_r.numpy())
                visitation_logs_c.append(visitation_log_c.numpy())

                # every call of generate_visitation_log changes rc_warp_dense; keep this one:
                if cam_r == preferred_camera[0] and cam_c == preferred_camera[1]:
                    rc_warp_dense_to_remember = self.rc_warp_dense.numpy()

        if rc_warp_dense_to_remember is not None:
            self.rc_warp_dense = rc_warp_dense_to_remember
            print('remembered rc_warp_dense for (cam_r, cam_c) = ' + str(preferred_camera))

        # THIRD, reshape visitation logs and return; shape should be (cam0, cam1) + visitation log shape;
        visitation_log_shape = self.camera_dims + visitation_logs_r[0].shape
        self.visitation_logs_r_for_all_cameras = np.stack(visitation_logs_r, axis=0).reshape(visitation_log_shape)
        self.visitation_logs_c_for_all_cameras = np.stack(visitation_logs_c, axis=0).reshape(visitation_log_shape)

        # convert to tf object:
        self.visitation_logs_r_for_all_cameras = tf.constant(self.visitation_logs_r_for_all_cameras)
        self.visitation_logs_c_for_all_cameras = tf.constant(self.visitation_logs_c_for_all_cameras)

        return self.visitation_logs_r_for_all_cameras, self.visitation_logs_c_for_all_cameras

    def generate_patched_dataset(self, num_patches, patch_size, patch_recon_size=None, sample_margin=None,
                                 fracture_big_tensors=False, prefetch=1, inclusive_patch_selection=False):
        # Does exactly what mcam3d's version does, but also generate np.indices, which depends on self.patch_size.
        # Generate dataset of patches from the image stack based on a selected patch in the reconstruction
        # dataset generated from self.visitation_log_r/c, self.rc_warp_dense, self.stack.
        # One element of a batch consists of 2-9 raw image patches (could be more, depending on the
        # mcam configuration) that are known to intersect at a given location in the reconstruction; thus, a batch is a
        # raggedtensor.
        # num_patch is basically the analog of batch_size for mcam3d (I don't want to override batch_size to preserve
        # functionality of regular stitching).
        # Run this function after define_network_and_camera_parameters.
        # patch_recon_size is the size of the tensor you're scatter_nd'ing the patches into; if not supplied, it will
        # default to patch_size*2.
        # sample_margin: how much along the border of the reconstruction to exclude from sampling; a fraction between 0
        # and 1, where 1 basically means the sampling area is 0.
        # fracture_big_tensors: dataset complains if stack and rc_warp_dense are too big (such as for 54 3000x4000 MCAM
        # datasets).
        # inclusive_patch_selection: when retrieving from the visitation log, check intersection with any point within
        # r:r+patch_size, c:c+patch_size (i.e., old behavior); otherwise, only check r,c.

        self.num_patches = num_patches
        self.patch_size = patch_size
        assert patch_size % 2 == 0  # we'll be dividing this by 2 in _gather_image_patches
        self._validate_patch_size()
        if patch_recon_size is None:
            self.patch_recon_size = 3 * self.patch_size
        else:
            self.patch_recon_size = patch_recon_size

        if fracture_big_tensors:
            self.fracture_size = 1
            self.stack_tf0 = self.stack_tf[0:1]
            self.rc_warp_dense0 = self.rc_warp_dense[0:1]
            self.stack_tf1 = self.stack_tf[1:2]
            self.rc_warp_dense1 = self.rc_warp_dense[1:2]
            self.stack_tf2 = self.stack_tf[2:3]
            self.rc_warp_dense2 = self.rc_warp_dense[2:3]
            self.stack_tf3 = self.stack_tf[3:4]
            self.rc_warp_dense3 = self.rc_warp_dense[3:4]
            self.stack_tf4 = self.stack_tf[4:5]
            self.rc_warp_dense4 = self.rc_warp_dense[4:5]
            self.stack_tf5 = self.stack_tf[5:6]
            self.rc_warp_dense5 = self.rc_warp_dense[5:6]
            self.stack_tf6 = self.stack_tf[6:7]
            self.rc_warp_dense6 = self.rc_warp_dense[6:7]
            self.stack_tf7 = self.stack_tf[7:8]
            self.rc_warp_dense7 = self.rc_warp_dense[7:8]
            self.stack_tf8 = self.stack_tf[8:9]
            self.rc_warp_dense8 = self.rc_warp_dense[8:9]
            self.stack_tf9 = self.stack_tf[9:10]
            self.rc_warp_dense9 = self.rc_warp_dense[9:10]
            self.stack_tf10 = self.stack_tf[10:11]
            self.rc_warp_dense10 = self.rc_warp_dense[10:11]
            self.stack_tf11 = self.stack_tf[11:12]
            self.rc_warp_dense11 = self.rc_warp_dense[11:12]
            self.stack_tf12 = self.stack_tf[12:13]
            self.rc_warp_dense12 = self.rc_warp_dense[12:13]
            self.stack_tf13 = self.stack_tf[13:14]
            self.rc_warp_dense13 = self.rc_warp_dense[13:14]
            self.stack_tf14 = self.stack_tf[14:15]
            self.rc_warp_dense14 = self.rc_warp_dense[14:15]
            self.stack_tf15 = self.stack_tf[15:16]
            self.rc_warp_dense15 = self.rc_warp_dense[15:16]
            self.stack_tf16 = self.stack_tf[16:17]
            self.rc_warp_dense16 = self.rc_warp_dense[16:17]
            self.stack_tf17 = self.stack_tf[17:18]
            self.rc_warp_dense17 = self.rc_warp_dense[17:18]
            self.stack_tf18 = self.stack_tf[18:19]
            self.rc_warp_dense18 = self.rc_warp_dense[18:19]
            self.stack_tf19 = self.stack_tf[19:20]
            self.rc_warp_dense19 = self.rc_warp_dense[19:20]
            self.stack_tf20 = self.stack_tf[20:21]
            self.rc_warp_dense20 = self.rc_warp_dense[20:21]
            self.stack_tf21 = self.stack_tf[21:22]
            self.rc_warp_dense21 = self.rc_warp_dense[21:22]
            self.stack_tf22 = self.stack_tf[22:23]
            self.rc_warp_dense22 = self.rc_warp_dense[22:23]
            self.stack_tf23 = self.stack_tf[23:24]
            self.rc_warp_dense23 = self.rc_warp_dense[23:24]
            self.stack_tf24 = self.stack_tf[24:25]
            self.rc_warp_dense24 = self.rc_warp_dense[24:25]
            self.stack_tf25 = self.stack_tf[25:26]
            self.rc_warp_dense25 = self.rc_warp_dense[25:26]
            self.stack_tf26 = self.stack_tf[26:27]
            self.rc_warp_dense26 = self.rc_warp_dense[26:27]
            self.stack_tf27 = self.stack_tf[27:28]
            self.rc_warp_dense27 = self.rc_warp_dense[27:28]
            self.stack_tf28 = self.stack_tf[28:29]
            self.rc_warp_dense28 = self.rc_warp_dense[28:29]
            self.stack_tf29 = self.stack_tf[29:30]
            self.rc_warp_dense29 = self.rc_warp_dense[29:30]
            self.stack_tf30 = self.stack_tf[30:31]
            self.rc_warp_dense30 = self.rc_warp_dense[30:31]
            self.stack_tf31 = self.stack_tf[31:32]
            self.rc_warp_dense31 = self.rc_warp_dense[31:32]
            self.stack_tf32 = self.stack_tf[32:33]
            self.rc_warp_dense32 = self.rc_warp_dense[32:33]
            self.stack_tf33 = self.stack_tf[33:34]
            self.rc_warp_dense33 = self.rc_warp_dense[33:34]
            self.stack_tf34 = self.stack_tf[34:35]
            self.rc_warp_dense34 = self.rc_warp_dense[34:35]
            self.stack_tf35 = self.stack_tf[35:36]
            self.rc_warp_dense35 = self.rc_warp_dense[35:36]
            self.stack_tf36 = self.stack_tf[36:37]
            self.rc_warp_dense36 = self.rc_warp_dense[36:37]
            self.stack_tf37 = self.stack_tf[37:38]
            self.rc_warp_dense37 = self.rc_warp_dense[37:38]
            self.stack_tf38 = self.stack_tf[38:39]
            self.rc_warp_dense38 = self.rc_warp_dense[38:39]
            self.stack_tf39 = self.stack_tf[39:40]
            self.rc_warp_dense39 = self.rc_warp_dense[39:40]
            self.stack_tf40 = self.stack_tf[40:41]
            self.rc_warp_dense40 = self.rc_warp_dense[40:41]
            self.stack_tf41 = self.stack_tf[41:42]
            self.rc_warp_dense41 = self.rc_warp_dense[41:42]
            self.stack_tf42 = self.stack_tf[42:43]
            self.rc_warp_dense42 = self.rc_warp_dense[42:43]
            self.stack_tf43 = self.stack_tf[43:44]
            self.rc_warp_dense43 = self.rc_warp_dense[43:44]
            self.stack_tf44 = self.stack_tf[44:45]
            self.rc_warp_dense44 = self.rc_warp_dense[44:45]
            self.stack_tf45 = self.stack_tf[45:46]
            self.rc_warp_dense45 = self.rc_warp_dense[45:46]
            self.stack_tf46 = self.stack_tf[46:47]
            self.rc_warp_dense46 = self.rc_warp_dense[46:47]
            self.stack_tf47 = self.stack_tf[47:48]
            self.rc_warp_dense47 = self.rc_warp_dense[47:48]
            self.stack_tf48 = self.stack_tf[48:49]
            self.rc_warp_dense48 = self.rc_warp_dense[48:49]
            self.stack_tf49 = self.stack_tf[49:50]
            self.rc_warp_dense49 = self.rc_warp_dense[49:50]
            self.stack_tf50 = self.stack_tf[50:51]
            self.rc_warp_dense50 = self.rc_warp_dense[50:51]
            self.stack_tf51 = self.stack_tf[51:52]
            self.rc_warp_dense51 = self.rc_warp_dense[51:52]
            self.stack_tf52 = self.stack_tf[52:53]
            self.rc_warp_dense52 = self.rc_warp_dense[52:53]
            self.stack_tf53 = self.stack_tf[53:54]
            self.rc_warp_dense53 = self.rc_warp_dense[53:54]
            self.stack_tf54 = self.stack_tf[54:55]
            self.rc_warp_dense54 = self.rc_warp_dense[54:55]
            self.stack_tf55 = self.stack_tf[55:56]
            self.rc_warp_dense55 = self.rc_warp_dense[55:56]
            self.stack_tf56 = self.stack_tf[56:57]
            self.rc_warp_dense56 = self.rc_warp_dense[56:57]
            self.stack_tf57 = self.stack_tf[57:58]
            self.rc_warp_dense57 = self.rc_warp_dense[57:58]
            self.stack_tf58 = self.stack_tf[58:59]
            self.rc_warp_dense58 = self.rc_warp_dense[58:59]
            self.stack_tf59 = self.stack_tf[59:60]
            self.rc_warp_dense59 = self.rc_warp_dense[59:60]
            self.stack_tf60 = self.stack_tf[60:61]
            self.rc_warp_dense60 = self.rc_warp_dense[60:61]
            self.stack_tf61 = self.stack_tf[61:62]
            self.rc_warp_dense61 = self.rc_warp_dense[61:62]
            self.stack_tf62 = self.stack_tf[62:63]
            self.rc_warp_dense62 = self.rc_warp_dense[62:63]
            self.stack_tf63 = self.stack_tf[63:64]
            self.rc_warp_dense63 = self.rc_warp_dense[63:64]
            self.stack_tf64 = self.stack_tf[64:65]
            self.rc_warp_dense64 = self.rc_warp_dense[64:65]
            self.stack_tf65 = self.stack_tf[65:66]
            self.rc_warp_dense65 = self.rc_warp_dense[65:66]
            self.stack_tf66 = self.stack_tf[66:67]
            self.rc_warp_dense66 = self.rc_warp_dense[66:67]
            self.stack_tf67 = self.stack_tf[67:68]
            self.rc_warp_dense67 = self.rc_warp_dense[67:68]
            self.stack_tf68 = self.stack_tf[68:69]
            self.rc_warp_dense68 = self.rc_warp_dense[68:69]
            self.stack_tf69 = self.stack_tf[69:70]
            self.rc_warp_dense69 = self.rc_warp_dense[69:70]
            self.stack_tf70 = self.stack_tf[70:71]
            self.rc_warp_dense70 = self.rc_warp_dense[70:71]
        else:
            self.fracture_size = None

        # run the network once so that we can access network.trainable_variables
        if self.use_postpended_channels_for_stitching:
            num_channels = self.num_channels - self.num_channels_rgb  # self.num_channels_rgb channels postpended, which
            # are for registration, not CNN input
        else:
            num_channels = self.num_channels
        out = self.network(tf.zeros([1, self.patch_size, self.patch_size, num_channels], dtype=self.tf_dtype))
        self.output_patch_size = out.numpy().shape[1]  # might not be same as self.patch_size if not using padded convs
        print('Output patch size: ' + str(self.output_patch_size))
        if self.patch_size != self.output_patch_size:
            print('Warning: training will work with output != input size, but inference on full images will not')

        if sample_margin is not None:
            row_low = sample_margin / 2 * self.recon_shape_base[0]
            row_high = (1 - sample_margin / 2) * self.recon_shape_base[0] - self.patch_size - 1
            col_low = sample_margin / 2 * self.recon_shape_base[1]
            col_high = (1 - sample_margin / 2) * self.recon_shape_base[1] - self.patch_size - 1
        else:
            row_low = 0
            row_high = self.recon_shape_base[0] - self.patch_size - 1
            col_low = 0
            col_high = self.recon_shape_base[1] - self.patch_size - 1

        # tf complains if this lambda function isn't defined on a standalone line:
        # (recon_shape_base is the non-downsampled size)
        generate_rand_coord = lambda x: (tf.random.uniform((), row_low, row_high),
                                         tf.random.uniform((), col_low, col_high))
        gather_image_patches = lambda r, c: self._gather_image_patches(r, c, inclusive_patch_selection)
        if inclusive_patch_selection:
            print('WARNING: inclusive_patch_selection behavior has changed and not been tested (01/09/2022)!')

        dataset = (tf.data.Dataset.range(1)  # dummy dataset
                   .map(generate_rand_coord)  # generate one random coordinate
                   .map(gather_image_patches)
                   .repeat(None)  # generate infinite number of patches
                   .apply(tf.data.experimental.dense_to_ragged_batch(batch_size=self.num_patches))  # different number
                   # of image patches per reconstruction patch
                   .prefetch(prefetch)
                   )  # ragged function seems to only work in tf2.3 (2.2 fails)

        self.d0, self.d1 = np.indices([self.patch_size, self.patch_size])  # for help with gather_nd later

        return dataset

    def _gather_image_patches(self, r, c, inclusive_patch_selection):
        # Same as mcam3d's, but different fracture size.
        # Used by generate_patched_dataset, but can also be used by user in eager mode for diagnostics.
        # Given r(ow) and c(olumn), corresponding to upper left corner, identify the image patches that overlap, based
        # on visitation_log; return patches from the raw image stack along with patches from the corresponding
        # rc_warp_dense coordinates.
        r = tf.cast(r * self.visitation_log_scale, dtype=tf.int32)
        c = tf.cast(c * self.visitation_log_scale, dtype=tf.int32)

        # retrieve records from visitation log:
        # remember that the visitation log has -1 for unvisited pixels!
        patch_size_scaled = tf.cast(self.patch_size * self.visitation_log_scale, dtype=tf.int32)
        if inclusive_patch_selection:
            retrieved_record_r = self.visitation_log_r[r:r + patch_size_scaled, c:c + patch_size_scaled]
            retrieved_record_c = self.visitation_log_c[r:r + patch_size_scaled, c:c + patch_size_scaled]
            # shapes: (patch_size, patch_size, num_images)
        else:
            r += patch_size_scaled // 2  # center the coordinate; sampling is based on upper left position
            c += patch_size_scaled // 2
            retrieved_record_r = self.visitation_log_r[r:r + 1, c:c + 1]
            retrieved_record_c = self.visitation_log_c[r:r + 1, c:c + 1]

        max_r = tf.reduce_max(retrieved_record_r, axis=(0, 1))  # shape: num_images
        max_c = tf.reduce_max(retrieved_record_c, axis=(0, 1))

        # if max_r/c is less than the max dim of images and greater than patch_size, then crop image from max-patch_size
        # to max; if max is less than patch_size, then crop image from 0 to patch_size; if max is greater than
        # image size, then crop from image_size-patch size to image_size

        # first, filter images by those which visited the current patch (if unvisited, max_r will be -1)
        inds_images_to_use = tf.cast(tf.where(max_r >= 0)[:, 0], tf.int32)  # length < num_images

        # use a for-loop here, at least to avoid doing a messy tf.gather operation
        im_patches = tf.TensorArray(tf.uint8, size=len(inds_images_to_use),  # use in lieu of list
                                    element_shape=(self.patch_size, self.patch_size, self.num_channels))
        rc_warp_patches = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                         element_shape=(self.patch_size, self.patch_size, 2))
        cam_to_vanish_batch = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                             element_shape=())
        vanish_point_batch = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                            element_shape=(2,))
        nominal_z_slices_batch = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                                element_shape=())
        inds_images_to_use_ = tf.TensorArray(tf.int32, size=len(inds_images_to_use),
                                         element_shape=())  # this may seem useless, but for some reason tf doesn't
        # combine inds_images_to_use into a ragged batch (at least in tf2.3); tf.zeros_like(inds_images_to_use) also
        # fails, but tf.zeros(len(inds_images_to_use)) succeeds

        for i in tf.range(len(inds_images_to_use)):  # can't use enumerate, or else tf might interpret as a python loop
            ind = inds_images_to_use[i]
            max_r_ind = max_r[ind]  # one number
            max_c_ind = max_c[ind]  # one number

            # three cases:
            # 1) max_r_ind < self.patch_size / 2
            # 2) max_r_ind >= self.stack.shape[1] - self.patch_size / 2
            # 3) self.patch_size <= max_r_ind < self.stack.shape[1] (treated as default below)
            # (and the same one for c)
            # note that patch_size is even
            r_start, r_end = tf.case([(tf.less(max_r_ind, self.patch_size // 2), lambda: (0, self.patch_size)),
                                      (tf.greater_equal(max_r_ind, self.stack.shape[1] - self.patch_size // 2),
                                       lambda: (self.stack.shape[1] - self.patch_size, self.stack.shape[1])),
                                      ], default=lambda: (max_r_ind - self.patch_size // 2,
                                                          max_r_ind + self.patch_size // 2))

            c_start, c_end = tf.case([(tf.less(max_c_ind, self.patch_size // 2), lambda: (0, self.patch_size)),
                                      (tf.greater_equal(max_c_ind, self.stack.shape[2] - self.patch_size // 2),
                                       lambda: (self.stack.shape[2] - self.patch_size, self.stack.shape[2])),
                                      ], default=lambda: (max_c_ind - self.patch_size // 2,
                                                          max_c_ind + self.patch_size // 2))

            if self.fracture_size is not None:
                fracture_num = tf.cast(ind / self.fracture_size, dtype=tf.int32)  # which fracture?
                ind_fracture = tf.math.floormod(ind, self.fracture_size)  # within that fracture, which index?
                stack_tf, rc_warp_dense = tf.switch_case(fracture_num, [lambda: (self.stack_tf0, self.rc_warp_dense0),
                                                                        lambda: (self.stack_tf1, self.rc_warp_dense1),
                                                                        lambda: (self.stack_tf2, self.rc_warp_dense2),
                                                                        lambda: (self.stack_tf3, self.rc_warp_dense3),
                                                                        lambda: (self.stack_tf4, self.rc_warp_dense4),
                                                                        lambda: (self.stack_tf5, self.rc_warp_dense5),
                                                                        lambda: (self.stack_tf6, self.rc_warp_dense6),
                                                                        lambda: (self.stack_tf7, self.rc_warp_dense7),
                                                                        lambda: (self.stack_tf8, self.rc_warp_dense8),
                                                                        lambda: (self.stack_tf9, self.rc_warp_dense9),
                                                                        lambda: (self.stack_tf10, self.rc_warp_dense10),
                                                                        lambda: (self.stack_tf11, self.rc_warp_dense11),
                                                                        lambda: (self.stack_tf12, self.rc_warp_dense12),
                                                                        lambda: (self.stack_tf13, self.rc_warp_dense13),
                                                                        lambda: (self.stack_tf14, self.rc_warp_dense14),
                                                                        lambda: (self.stack_tf15, self.rc_warp_dense15),
                                                                        lambda: (self.stack_tf16, self.rc_warp_dense16),
                                                                        lambda: (self.stack_tf17, self.rc_warp_dense17),
                                                                        lambda: (self.stack_tf18, self.rc_warp_dense18),
                                                                        lambda: (self.stack_tf19, self.rc_warp_dense19),
                                                                        lambda: (self.stack_tf20, self.rc_warp_dense20),
                                                                        lambda: (self.stack_tf21, self.rc_warp_dense21),
                                                                        lambda: (self.stack_tf22, self.rc_warp_dense22),
                                                                        lambda: (self.stack_tf23, self.rc_warp_dense23),
                                                                        lambda: (self.stack_tf24, self.rc_warp_dense24),
                                                                        lambda: (self.stack_tf25, self.rc_warp_dense25),
                                                                        lambda: (self.stack_tf26, self.rc_warp_dense26),
                                                                        lambda: (self.stack_tf27, self.rc_warp_dense27),
                                                                        lambda: (self.stack_tf28, self.rc_warp_dense28),
                                                                        lambda: (self.stack_tf29, self.rc_warp_dense29),
                                                                        lambda: (self.stack_tf30, self.rc_warp_dense30),
                                                                        lambda: (self.stack_tf31, self.rc_warp_dense31),
                                                                        lambda: (self.stack_tf32, self.rc_warp_dense32),
                                                                        lambda: (self.stack_tf33, self.rc_warp_dense33),
                                                                        lambda: (self.stack_tf34, self.rc_warp_dense34),
                                                                        lambda: (self.stack_tf35, self.rc_warp_dense35),
                                                                        lambda: (self.stack_tf36, self.rc_warp_dense36),
                                                                        lambda: (self.stack_tf37, self.rc_warp_dense37),
                                                                        lambda: (self.stack_tf38, self.rc_warp_dense38),
                                                                        lambda: (self.stack_tf39, self.rc_warp_dense39),
                                                                        lambda: (self.stack_tf40, self.rc_warp_dense40),
                                                                        lambda: (self.stack_tf41, self.rc_warp_dense41),
                                                                        lambda: (self.stack_tf42, self.rc_warp_dense42),
                                                                        lambda: (self.stack_tf43, self.rc_warp_dense43),
                                                                        lambda: (self.stack_tf44, self.rc_warp_dense44),
                                                                        lambda: (self.stack_tf45, self.rc_warp_dense45),
                                                                        lambda: (self.stack_tf46, self.rc_warp_dense46),
                                                                        lambda: (self.stack_tf47, self.rc_warp_dense47),
                                                                        lambda: (self.stack_tf48, self.rc_warp_dense48),
                                                                        lambda: (self.stack_tf49, self.rc_warp_dense49),
                                                                        lambda: (self.stack_tf50, self.rc_warp_dense50),
                                                                        lambda: (self.stack_tf51, self.rc_warp_dense51),
                                                                        lambda: (self.stack_tf52, self.rc_warp_dense52),
                                                                        lambda: (self.stack_tf53, self.rc_warp_dense53),
                                                                        lambda: (self.stack_tf54, self.rc_warp_dense54),
                                                                        lambda: (self.stack_tf55, self.rc_warp_dense55),
                                                                        lambda: (self.stack_tf56, self.rc_warp_dense56),
                                                                        lambda: (self.stack_tf57, self.rc_warp_dense57),
                                                                        lambda: (self.stack_tf58, self.rc_warp_dense58),
                                                                        lambda: (self.stack_tf59, self.rc_warp_dense59),
                                                                        lambda: (self.stack_tf60, self.rc_warp_dense60),
                                                                        lambda: (self.stack_tf61, self.rc_warp_dense61),
                                                                        lambda: (self.stack_tf62, self.rc_warp_dense62),
                                                                        lambda: (self.stack_tf63, self.rc_warp_dense63),
                                                                        lambda: (self.stack_tf64, self.rc_warp_dense64),
                                                                        lambda: (self.stack_tf65, self.rc_warp_dense65),
                                                                        lambda: (self.stack_tf66, self.rc_warp_dense66),
                                                                        lambda: (self.stack_tf67, self.rc_warp_dense67),
                                                                        lambda: (self.stack_tf68, self.rc_warp_dense68),
                                                                        lambda: (self.stack_tf69, self.rc_warp_dense69),
                                                                        lambda: (self.stack_tf70, self.rc_warp_dense70),
                                                                        ])
                rc_warp_patches = rc_warp_patches.write(i, rc_warp_dense[ind_fracture, r_start:r_end, c_start:c_end, :])
                im_patches = im_patches.write(i, stack_tf[ind_fracture, r_start:r_end, c_start:c_end, :])
            else:
                # no fracturing; index into the whole tensor:
                rc_warp_patches = rc_warp_patches.write(i, self.rc_warp_dense[ind, r_start:r_end, c_start:c_end, :])
                im_patches = im_patches.write(i, self.stack_tf[ind, r_start:r_end, c_start:c_end, :])
                # tf complains if I use self.stack (numpy version) rather than self.stack_tf (tf version)^

            # these variables are small; no need to give the fracture treatment
            inds_images_to_use_ = inds_images_to_use_.write(i, ind)
            vanish_point_batch = vanish_point_batch.write(i, self.vanish_point[ind])
            cam_to_vanish_batch = cam_to_vanish_batch.write(i, self.cam_to_vanish[ind])
            nominal_z_slices_batch = nominal_z_slices_batch.write(i, self.nominal_z_slices[ind])

        return (im_patches.stack(),  # shape: _ by patch_size by patch_size by 3
                rc_warp_patches.stack(),  # shape: _ by patch_size by patch_size by 2
                vanish_point_batch.stack(),  # shape: _ by 2
                cam_to_vanish_batch.stack(),  # shape: _
                nominal_z_slices_batch.stack(),  # shape: _
                inds_images_to_use_.stack(),  # shape: _
                r, c)  # also return the random coordinate

    def define_network_and_camera_params(self, vanish_point, cam_to_vanish, num_channels_rgb, filters_list=[16]*5,
                                         skip_list=[0]*5, learning_rate=1e-3, architecture='fcnn', optimizer=None):
        # Does exactly what mcam3d's version does, but replaces cam_to_vanish with nominal_z_slices.
        super().define_network_and_camera_params(vanish_point, cam_to_vanish, num_channels_rgb, filters_list,
                                                 skip_list, learning_rate, architecture, optimizer)
        assert self.nominal_z_slices.shape == self.cam_to_vanish.shape
        self.nominal_z_slices_copy = self.nominal_z_slices  # for plotting

    def _backproject_and_predict(self, im_patches, rc_warp_patches, vanish_point_batch, cam_to_vanish_batch,
                                 nominal_z_slices_batch,
                                 inds_images_to_use, r, c, stop_gradient=True, dither_coords=False,
                                 downsample_factor=1, use_hpf_for_MSE_loss=False, orthorectify=False):
        # Similar to the version in mcam3d, but we don't need to warp the coordinates further.
        # Generate camera-centric z-stack indices, backproject to get reconstruction.
        # Input arguments are from tf.dataset; some are not needed for zstitch.
        # Specifically, unpack the ragged batches (effectively flattening along the ragged dimension) and use row_splits
        # to keep track of the batch boundaries (or better yet, value_rowids(), which gives me the indices for batch
        # membership, which I can use for scatter_nd).
        # Note that num_channels and num_channels_rgb should always be the same for zstitch, barring future changes to
        # allow for additional channels to input to the CNN.
        # downsample_factor: downsample the patched reconstruction (for multi-resolution optimization).
        # use_hpf_for_MSE_loss: instead of comparing photometric values, compare high-pass-filtered values.
        # orthorectify: if True, then instead of averaging the nearest slices, pick one and orthorectify.

        patch_recon_size = tf.cast(self.patch_recon_size / downsample_factor, dtype=tf.int32)
        d0 = self.d0[::downsample_factor, ::downsample_factor]
        d1 = self.d1[::downsample_factor, ::downsample_factor]

        # unpack batch:
        im_flat = tf.cast(im_patches.values, self.tf_dtype)  # flattens ragged dimension
        # new shape^: _, patch, patch, channels
        partitions = tf.cast(im_patches.value_rowids(), tf.int32)  # shape: _
        rc_warp_flat = rc_warp_patches.values  # shape: _, patch, patch, 2
        vanish_point_flat = vanish_point_batch.values  # shape: _, 2  # not used (unless orthorectify == True!)
        cam_to_vanish_flat = cam_to_vanish_batch.values  # shape: _
        nominal_z_slices_flat = nominal_z_slices_batch.values  # shape: _

        # generate height map:
        if self.recompute_CNN:
            network = tf.recompute_grad(self.network)
        else:
            network = self.network

        CNN_input = im_flat
        fcnn_out = network(CNN_input)

        depth = tf.reduce_mean(fcnn_out, [-1]) * self.unet_scale  # depth prediction; integrate out channel dimension
        # shape: _, patch_size, patch_size

        # crop input if necessary:
        if self.patch_size > self.output_patch_size:
            # need to crop the patches from the input
            margin = (self.patch_size - self.output_patch_size) // 2
            rc_warp_flat = rc_warp_flat[:, margin:-margin, margin:-margin, :]
            im_flat = im_flat[:, margin:-margin, margin:-margin, :]

        # linearly interpolate depth value for indexing into stack
        # restrict to within range; casting is a flooring operation:
        num_z = self.num_channels
        assert self.num_channels == self.num_channels_rgb
        depth_index_float = tf.math.sin(depth) * (num_z / 2 - .51) + num_z / 2 - .5  # ensure within num_z range
        depth_index_floor = tf.cast(depth_index_float, dtype=tf.int32)
        depth_index_ceil = depth_index_floor + 1
        dist2floor = depth_index_float - tf.cast(depth_index_floor, dtype=tf.float32)
        dist2ceil = 1 - dist2floor
        # ^these are all _ x patch_size x patch_size

        if orthorectify:
            depth_index_round = tf.round(depth_index_float)
            residual_height = (depth_index_float - depth_index_round) * self.z_step_mm
            if self.z_stage_up:
                residual_height = -residual_height
            f_eff = self.effective_focal_length_mm
            M_j = self.magnification_j
            H = cam_to_vanish_flat[:, None, None]

            r = rc_warp_flat - vanish_point_flat[:, None, None, :]  # lateral distance to vanishing point
            delta_radial = residual_height / f_eff / (
                    1 + 1 / M_j * H / self.H_j)  # radial deform field based on height map
            rc_warp_flat = r * (1 - delta_radial[:, :, :, None]) + vanish_point_flat[:, None, None,
                                                                   :]  # add back vanishing point
            # shape of rc_warp_flat: _, patch, patch, 2

            depth_index_round = tf.cast(depth_index_round, dtype=tf.int32)  # cast for later

        # Retrieve the values corresponding to the best per-pixel depths:
        # im_flat shape: _,patch,patch,num_z
        # depth_index_floor/ceil are of shape: _,patch,patch
        # The following will output something of shape _,patch,patch, using the latter to pick out from one depth of
        # each pixel of the former; use map_fn to do this for each element of the batch.
        fn = lambda x: tf.gather_nd(x[0], tf.stack([d0, d1, x[1]], axis=2))
        if not use_hpf_for_MSE_loss:
            if not orthorectify:
                im_all_in_focus_floor = tf.map_fn(fn=fn, elems=(im_flat, depth_index_floor), fn_output_signature=tf.float32)
                im_all_in_focus_ceil = tf.map_fn(fn=fn, elems=(im_flat, depth_index_ceil), fn_output_signature=tf.float32)

                # linear interpolation:
                im_all_in_focus_predict = im_all_in_focus_floor * dist2ceil + im_all_in_focus_ceil * dist2floor
            else:
                # pick the value corresponding to the one you're orthorectifying:
                im_all_in_focus_predict = tf.map_fn(fn=fn, elems=(im_flat, depth_index_round),
                                                    fn_output_signature=tf.float32)

        else:
            pass  # handle below (or move this if-else statement below)

        # compute sharpness of z stack:
        im_flat_T = tf.transpose(im_flat, [3, 0, 1, 2])  # put z-stack dimension first so it can be batch dim for conv
        # ^now, it's of shape z, _, patch_size, patch_size
        blurred = tf.nn.conv2d(im_flat_T[:, :, :, :, None],  # the leading dims are batch dims; add dummy channels dim
                               self.gaussian_kernel[:, :, None, None],
                               strides=1, padding='SAME')[:, :, :, :, 0]  # remove channel dim
        im_flat_hpf = im_flat_T / blurred  # sort of high-pass-filter, but normalized
        dx = tf.nn.conv2d(im_flat_hpf[:, :, :, :, None],  # derivative along first dim
                          self.diff0_kernel[:, :, None, None],
                          strides=1, padding='SAME')[:, :, :, :, 0]
        dy = tf.nn.conv2d(im_flat_hpf[:, :, :, :, None],  # derivative along second dim
                          self.diff1_kernel[:, :, None, None],
                          strides=1, padding='SAME')[:, :, :, :, 0]
        dxy = tf.sqrt(dx ** 2 + dy ** 2)  # ~TV
        dxy_blur = tf.nn.conv2d(dxy[:, :, :, :, None],  # blur again
                                self.gaussian_kernel[:, :, None, None],
                                strides=1, padding='SAME')[:, :, :, :, 0]
        dxy_blur = tf.transpose(dxy_blur, [1, 2, 3, 0])  # move z-stack dimension back to the end; _, patch, patch, z
        dxy_blur_floor = tf.map_fn(fn=fn, elems=(dxy_blur, depth_index_floor), fn_output_signature=tf.float32)
        dxy_blur_ceil = tf.map_fn(fn=fn, elems=(dxy_blur, depth_index_ceil), fn_output_signature=tf.float32)
        dxy_predict = dxy_blur_floor * dist2ceil + dxy_blur_ceil * dist2floor  # 'all-in-focus' sharpness
        if self.weighted_sharpness_loss:
            max_ = tf.reduce_max(dxy_blur, axis=-1)
            weight = tf.maximum(4 * (max_ / 0.0375 - self.weighted_sharpness_thresholds[0]), 0)  # arbitrary dividend and scale
            weighted_sharpness = dxy_predict * weight
            weighted_sharpness = weighted_sharpness[:, 2 * self.sigma:-2 * self.sigma,  # to avoid edge conv effects
                                                       2 * self.sigma:-2 * self.sigma]
            self.sharpness_batch = tf.reduce_mean(weighted_sharpness)

            # weighted argmax MSE:
            argmax_ = tf.math.argmax(dxy_blur, axis=-1)  # shape: _, patch, patch
            argmax_ = tf.cast(argmax_, dtype=tf.float32)
            harsher_weight = tf.maximum(4 * (max_ / 0.0375 - self.weighted_sharpness_thresholds[1]), 0)
            weighted_errors = harsher_weight * (argmax_ - depth_index_float) ** 2
            weighted_errors = weighted_errors[:, 2 * self.sigma:-2 * self.sigma,  # to avoid edge conv effects
                                                 2 * self.sigma:-2 * self.sigma]
            self.MSE_argmax = tf.reduce_mean(weighted_errors)

        else:
            self.sharpness_batch = tf.reduce_mean(dxy_predict)  # new loss term

            # another potential loss term based on argmax supervision:
            argmax_ = tf.math.argmax(dxy_blur, axis=-1)  # shape: _, patch, patch
            argmax_ = tf.cast(argmax_, dtype=tf.float32)
            self.MSE_argmax = tf.reduce_mean((argmax_ - depth_index_float) ** 2)

        if use_hpf_for_MSE_loss:
            blurred1 = tf.nn.conv2d(im_flat_T[:, :, :, :, None],
                                    # the leading dims are batch dims; add dummy channels dim
                                    self.gaussian_kernel2_sqrt2[:, :, None, None],
                                    strides=1, padding='SAME')[:, :, :, :, 0]  # remove channel dim
            blurred2 = tf.nn.conv2d(im_flat_T[:, :, :, :, None],
                                    self.gaussian_kernel2[:, :, None, None],
                                    strides=1, padding='SAME')[:, :, :, :, 0]
            im_flat_hpf = blurred1 - blurred2

            im_flat_hpf = tf.transpose(im_flat_hpf,
                                       [1, 2, 3, 0])  # move z-stack dimension back to the end; _, patch, patch, z
            if not orthorectify:
                im_all_in_focus_floor = tf.map_fn(fn=fn, elems=(im_flat_hpf, depth_index_floor), fn_output_signature=tf.float32)
                im_all_in_focus_ceil = tf.map_fn(fn=fn, elems=(im_flat_hpf, depth_index_ceil), fn_output_signature=tf.float32)
                im_all_in_focus_predict = im_all_in_focus_floor * dist2ceil + im_all_in_focus_ceil * dist2floor
            else:
                # pick the value corresponding to the one you're orthorectifying:
                im_all_in_focus_predict = tf.map_fn(fn=fn, elems=(im_flat_hpf, depth_index_round),
                                                    fn_output_signature=tf.float32)

        # flatten and stack:
        im_all_in_focus_predict = tf.reshape(im_all_in_focus_predict, [-1, self.patch_size**2])
        dxy_predict = tf.reshape(dxy_predict, [-1, self.patch_size**2])
        depth_index_float = tf.reshape(depth_index_float, [-1, self.patch_size**2])
        if self.z_stage_up:
            depth_index_float = depth_index_float - nominal_z_slices_flat[:, None] + num_z / 2  # sync to the reference
        else:
            depth_index_float = depth_index_float + nominal_z_slices_flat[:, None] - num_z / 2  # sync to the reference
        # ^can add a constant to this, and shouldn't make a difference, because error is in differences
        self.im = tf.stack([im_all_in_focus_predict, depth_index_float], axis=2)  # add height as 2nd channel

        # rc_warp doesn't need further warping;
        rc = tf.reshape(rc_warp_flat, [-1, self.patch_size ** 2, 2])

        if dither_coords:
            # random rotation:
            theta = tf.random.uniform((), 0, 2 * np.pi, dtype=self.tf_dtype)
            cos = tf.cos(theta)
            sin = tf.sin(theta)
            rotmat = tf.stack([[cos, sin], [-sin, cos]])
            rc = tf.einsum('abc,cd->abd', rc, rotmat)
            # random sub-pixel translation:
            rc = rc + tf.random.uniform([1, 1, 2], -1, 1, dtype=self.tf_dtype)

        rc = rc / downsample_factor

        # backprojection coordinate generation, as usual:
        # neighboring pixels:
        rc_floor = tf.floor(rc)
        rc_ceil = rc_floor + 1

        # distance to neighboring pixels:
        frc = rc - rc_floor
        crc = rc_ceil - rc

        # cast
        rc_floor = tf.cast(rc_floor, tf.int32)
        rc_ceil = tf.cast(rc_ceil, tf.int32)

        # force the use of mod as the restrict function for dealing with out-of-bounds coordinates; this means that if
        # you make the patch recon large, the code will be tolerant to errors in centering the patches
        restrict = lambda x: tf.math.floormod(x, patch_recon_size)

        self.rc_ff = restrict(rc_floor)
        self.rc_cc = restrict(rc_ceil)
        self.rc_cf = restrict(tf.stack([rc_ceil[:, :, 0], rc_floor[:, :, 1]], 2))
        self.rc_fc = restrict(tf.stack([rc_floor[:, :, 0], rc_ceil[:, :, 1]], 2))

        self.frc = tf.exp(-frc ** 2 / 2. / self.sig_proj ** 2)
        self.crc = tf.exp(-crc ** 2 / 2. / self.sig_proj ** 2)

        # augmented coordinates:
        rc_4 = tf.stack([self.rc_ff, self.rc_cc, self.rc_cf, self.rc_fc], 0)  # shape: 4, _, patch**2, 2
        rcp_4 = tf.concat([rc_4, tf.broadcast_to(partitions[None, :, None, None],  # shape: 4, _, patch**2, 3
                                                (4, len(partitions), self.output_patch_size ** 2, 1))], 3)
        rcp_4 = tf.reshape(rcp_4, [-1, 3])  # finally, flatten

        # interpolated:
        im_4 = tf.stack([self.im * self.frc[:, :, 0, None] * self.frc[:, :, 1, None],
                         self.im * self.crc[:, :, 0, None] * self.crc[:, :, 1, None],
                         self.im * self.crc[:, :, 0, None] * self.frc[:, :, 1, None], # shape: 4, _, patch**2, channels
                         self.im * self.frc[:, :, 0, None] * self.crc[:, :, 1, None]], 0)
        w_4 = tf.stack([self.frc[:, :, 0] * self.frc[:, :, 1],
                        self.crc[:, :, 0] * self.crc[:, :, 1],
                        self.crc[:, :, 0] * self.frc[:, :, 1],
                        self.frc[:, :, 0] * self.crc[:, :, 1]], 0)  # shape: 4, _, patch**2
        im_4 = tf.reshape(im_4, [-1, 2])  # 2: all-in-focus image, and depth
        w_4= tf.reshape(w_4,[-1])

        # backproject:
        self.normalize = tf.scatter_nd(rcp_4, w_4, [patch_recon_size, patch_recon_size, self.num_patches])
        self.recon = tf.scatter_nd(rcp_4, im_4, [patch_recon_size, patch_recon_size,
                                                 self.num_patches, 2])  # 2: all-in-focus image, and depth
        self.recon = tf.math.divide_no_nan(self.recon, self.normalize[:, :, :, None])
        # shape: patch_recon_size, patch_recon_size, num patches, num channels

        if stop_gradient:
            self.recon = tf.stop_gradient(self.recon)

        # now, forward prediction:
        gathered = tf.gather_nd(self.recon, rcp_4)  # shape: 4*_*patch*patch, channels
        gathered = tf.reshape(gathered, (4, -1, self.output_patch_size ** 2, 2))  # 2: all-in-focus image, and depth
        ff, cc, cf, fc = tf.unstack(gathered, num=4, axis=0)  # shape of each: _, patch*patch, channels

        self.forward = (ff * self.frc[:, :, 0, None] * self.frc[:, :, 1, None] +
                        cc * self.crc[:, :, 0, None] * self.crc[:, :, 1, None] +
                        cf * self.crc[:, :, 0, None] * self.frc[:, :, 1, None] +
                        fc * self.frc[:, :, 0, None] * self.crc[:, :, 1, None])

        self.forward /= ((self.frc[:, :, 0, None] * self.frc[:, :, 1, None]) +
                         (self.crc[:, :, 0, None] * self.crc[:, :, 1, None]) +
                         (self.crc[:, :, 0, None] * self.frc[:, :, 1, None]) +
                         (self.frc[:, :, 0, None] * self.crc[:, :, 1, None]))  # shape: _, patch**2, channels

        # error between prediction and data:
        # split off the last dimension, the height dimension, to compute the height map MSE:
        self.forward_height = self.forward[:, :, -1]
        self.error_height = self.forward_height - self.im[:, :, -1]
        self.error = self.forward[:, :, :-1] - self.im[:, :, :-1]  # remaining channels are the actual recon

        if self.TV_relaxation_coeff is None:
            self.MSE_height = tf.reduce_mean(self.error_height ** 2)
            self.MSE = tf.reduce_mean(self.error ** 2)
            self.loss_weight = None
        else:
            height = self.recon[:, :, :, -1]
            d0 = height[1:, :-1] - height[:-1, :-1]
            d1 = height[:-1, 1:] - height[:-1, :-1]
            self.TV2 = d0 ** 2 + d1 ** 2
            self.TV2 = tf.stop_gradient(self.TV2)
            loss_weight = tf.gather_nd(self.TV2, rcp_4)
            self.loss_weight = tf.reshape(loss_weight,
                                          (4, -1, self.output_patch_size ** 2))[0]  # pick one of the 4 pixels
            self.loss_weight = tf.exp(-self.TV_relaxation_coeff * self.loss_weight)
            self.MSE = tf.reduce_mean(self.loss_weight[:, :, None] * self.error ** 2)
            self.MSE_height = tf.reduce_mean(self.loss_weight * self.error_height ** 2)
            self.tensors_to_track['loss_weight'] = self.loss_weight

        return self.recon, self.normalize, self.forward, self.loss_weight

    @tf.function
    def gradient_update_patch(self, batch, height_map_reg_coef=None, return_tracked_tensors=False, stop_gradient=True,
                              return_loss_only=False, return_gradients=False, clip_gradient_norm=None,
                              dither_coords=False, downsample_factor=1, sharpness_reg_coef=None, stitch_loss_coef=1,
                              argmax_loss_coef=.01, use_hpf_for_MSE_loss=False, orthorectify=False
                              ):
        # Slightly modified version of mcam3d's to accommodate sharpness regularization, with the option to turn off
        # the stitching-based loss. The stitching loss coefficient can also be tuned.
        # clip_gradient_norm: pick a threshold to clip to (tf.clip_by_norm);
        with tf.GradientTape() as tape:
            self._backproject_and_predict(*batch, stop_gradient, dither_coords, downsample_factor,
                                          use_hpf_for_MSE_loss, orthorectify)

            loss_list = list()
            if stitch_loss_coef is not None:
                loss_list.append(stitch_loss_coef * self.MSE)
            if height_map_reg_coef is not None:
                loss_list.append(height_map_reg_coef * self.MSE_height)
            if sharpness_reg_coef is not None:
                loss_list.append(-sharpness_reg_coef * self.sharpness_batch)
            if argmax_loss_coef is not None:
                loss_list.append(argmax_loss_coef * self.MSE_argmax)

            loss = tf.reduce_sum(loss_list)

        grads = tape.gradient(loss, self.network.trainable_variables)
        if clip_gradient_norm is not None:
            grads, global_norm = tf.clip_by_global_norm(grads, clip_gradient_norm)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        if return_loss_only:
            return_list = [loss_list]
        else:
            if return_tracked_tensors:
                return_list = [loss_list, self.recon, self.normalize, self.tensors_to_track]
            else:
                return_list = [loss_list, self.recon, self.normalize]

        if return_gradients:
            return_list.append(grads)
        if clip_gradient_norm is not None:
            return_list.append(global_norm)

        return return_list

    def generate_full_recon(self, margin=None, stitch_rgb=True, new_stack=None, inds_keep=None,
                            compute_confidence_map=False):
        # Modified version of mcam3d's so that it uses z-stacks.
        # Run the network on each image in a for loop, and backproject sequentially.
        # May be a good idea to do this on CPU.
        # Adapted from backproject_and_predict and _generate_recon.
        # margin: a value in pixels that specifies how much to crop the output of the CNN to remove edge effects.
        # stitch_rgb: only relevant when self.use_postpended_channels_for_stitching is True; instead of reconstructing
        # with the postpended channels, as is done during training, reconstruct using the rgb channels, since we just
        # want a nice forward prediction.
        # new_stack: if you want to use a different dataset stack than what was used for optimization.
        # inds_keep: if you want to not use all cameras to generate full FOV.
        # compute_confidence_map: instead of generating all-in-focus image, generate the confidence map.

        if new_stack is None:
            stack = self.stack
        else:
            assert self.stack.shape == new_stack.shape
            stack = new_stack

        # accumulate these tensors with the for loop:
        recon_cumulative = tf.zeros(list(self.recon_shape_base) + [2], dtype=self.tf_dtype)  # 2: all-in-focus image, and depth
        normalize_cumulative = tf.zeros(self.recon_shape_base, dtype=self.tf_dtype)

        # create padding and depadding layers (this should only ever be used for full reconstruction generation):
        self.padded_shape = [self.network.get_compatible_size(dim) for dim in stack.shape[1:3]]
        pad_r = self.padded_shape[0] - stack.shape[1]
        pad_c = self.padded_shape[1] - stack.shape[2]
        pad_top = pad_r // 2
        pad_bottom = int(tf.math.ceil(pad_r / 2))
        pad_left = pad_c // 2
        pad_right = int(tf.math.ceil(pad_c / 2))
        pad_specs = ((pad_top, pad_bottom), (pad_left, pad_right))
        pad_layer = tf.keras.layers.ZeroPadding2D(pad_specs)
        depad_layer = tf.keras.layers.Cropping2D(pad_specs)

        restrict = lambda x: tf.math.floormod(x, self.recon_shape_base)

        d0, d1 = np.indices(np.array(stack.shape[1:3]) - margin*2)  # for help with gathering below

        if inds_keep is not None:
            stack_ = tf.gather(stack, inds_keep)
            rc_warp_dense_ = tf.gather(self.rc_warp_dense, inds_keep)
            nominal_z_slices_ = tf.gather(self.nominal_z_slices, inds_keep)
        else:
            stack_ = stack
            rc_warp_dense_ = self.rc_warp_dense
            nominal_z_slices_ = self.nominal_z_slices_copy

        for im, rc_warp, nominal_z_slice in tqdm(zip(stack_, rc_warp_dense_, nominal_z_slices_)):
            # im shape: row, col, num_dim
            # rc_warp shape: row, col, 2
            # nominal_z_slice shape: ()

            im = tf.cast(im, dtype=self.tf_dtype)[None]  # cast from uint8 to float32; add batch dim

            if compute_confidence_map:
                # compute sharpness of whole stack:
                raise Exception('confidence map not yet implemented')
            else:
                CNN_input = im

                # generate height map:
                im_pad = pad_layer(CNN_input)  # pad to a shape the network likes
                im = im[0]  # don't need batch dim anymore
                fcnn_out = self.network(im_pad)
                fcnn_depad = depad_layer(fcnn_out)[0]  # depad, and remove batch dimension

                if margin is not None:
                    fcnn_depad = fcnn_depad[margin:-margin, margin:-margin, :]
                    rc_warp = rc_warp[margin:-margin, margin:-margin, :]
                    im = im[margin:-margin, margin:-margin, :]

                depth = tf.reduce_mean(fcnn_depad, [-1]) * self.unet_scale  # depth prediction; remove feature dimension

                # linearly interpolate depth value for indexing into stack and restrict to within range:
                num_z = self.num_channels
                assert self.num_channels == self.num_channels_rgb
                depth_index_float = tf.math.sin(depth) * (num_z / 2 - .51) + num_z / 2 - .5
                depth_index_floor = tf.cast(depth_index_float, dtype=tf.int32)
                depth_index_ceil = depth_index_floor + 1
                dist2floor = depth_index_float - tf.cast(depth_index_floor, dtype=tf.float32)
                dist2ceil = 1 - dist2floor
                # ^these are all row x col

                im_all_in_focus_floor = tf.gather_nd(im, tf.stack([d0, d1, depth_index_floor], axis=2))
                im_all_in_focus_ceil = tf.gather_nd(im, tf.stack([d0, d1, depth_index_ceil], axis=2))

                im_all_in_focus_predict = im_all_in_focus_floor * dist2ceil + im_all_in_focus_ceil * dist2floor

                # flatten out spatial dims (batch dim is 1):
                depth = tf.reshape(depth, [-1])
                rc_warp = tf.reshape(rc_warp, [-1, 2])
                im_all_in_focus_predict = tf.reshape(im_all_in_focus_predict, [-1])
                depth_index_float = tf.reshape(depth_index_float, [-1])
                if self.z_stage_up:
                    depth_index_float = depth_index_float - nominal_z_slice + num_z / 2  # sync to reference (flat target)
                else:
                    depth_index_float = depth_index_float + nominal_z_slice - num_z / 2  # sync to reference (flat target)

                # stacking:
                self.im = tf.stack([im_all_in_focus_predict, depth_index_float], axis=1)  # add height as 2nd channel

            # rc_warp doesn't need further warping
            rc = rc_warp
            # shape of rc: _, 2

            # backprojection coordinate generation, as usual:
            # neighboring pixels:
            rc_floor = tf.floor(rc)
            rc_ceil = rc_floor + 1

            # distance to neighboring pixels:
            frc = rc - rc_floor
            crc = rc_ceil - rc

            # cast
            rc_floor = tf.cast(rc_floor, tf.int32)
            rc_ceil = tf.cast(rc_ceil, tf.int32)

            self.rc_ff = restrict(rc_floor)
            self.rc_cc = restrict(rc_ceil)
            self.rc_cf = restrict(tf.stack([rc_ceil[:, 0], rc_floor[:, 1]], 1))
            self.rc_fc = restrict(tf.stack([rc_floor[:, 0], rc_ceil[:, 1]], 1))

            self.frc = tf.exp(-frc ** 2 / 2. / self.sig_proj ** 2)
            self.crc = tf.exp(-crc ** 2 / 2. / self.sig_proj ** 2)

            # augmented coordinates:
            rc_4 = tf.concat([self.rc_ff, self.rc_cc, self.rc_cf, self.rc_fc], 0)

            # interpolated:
            im_4 = tf.concat([self.im * self.frc[:, 0, None] * self.frc[:, 1, None],
                              self.im * self.crc[:, 0, None] * self.crc[:, 1, None],
                              self.im * self.crc[:, 0, None] * self.frc[:, 1, None],
                              self.im * self.frc[:, 0, None] * self.crc[:, 1, None]], 0)
            w_4 = tf.concat([self.frc[:, 0] * self.frc[:, 1],
                             self.crc[:, 0] * self.crc[:, 1],
                             self.crc[:, 0] * self.frc[:, 1],
                             self.frc[:, 0] * self.crc[:, 1]], 0)
            # backproject:
            normalize = tf.scatter_nd(rc_4, w_4, self.recon_shape_base)
            recon = tf.scatter_nd(rc_4, im_4, [self.recon_shape_base[0], self.recon_shape_base[1],
                                               2])  # 2: all-in-focus image, and depth
            recon_cumulative = recon_cumulative + recon
            normalize_cumulative = normalize_cumulative + normalize

        recon = tf.math.divide_no_nan(recon_cumulative, normalize_cumulative[:, :, None])

        return recon, normalize_cumulative


def get_z_step_mm(directory):
    # given a directory of all the MCAM data files, return the z-stack by checking one of them

    import glob
    for any_file_will_do in glob.glob(directory + '/**/z_stack_*', recursive=True):
        break
    data = xr.open_dataset(any_file_will_do)
    z_step_mm = np.mean(np.diff(data.z_stage)) * 1000  # z step in mm

    return z_step_mm
