import numpy as np
import tensorflow as tf
import cv2
from tqdm.notebook import tqdm
import xarray as xr
from pathlib import Path
import scipy.io
from zstitch import zstitch
from mcam3d import fcnn
from mcam_loading_scripts import filepath_key
import h5py


class gmosaic:
    def __init__(self, filepath, restore_path_camera_params, sensor_crop=(3072, 3072), camera_dims=(9, 6),
                 z_index_ref=None, grayscale=False, cam_slice0=None, cam_slice1=None, batch_size=1,
                 rc_global_shift=(10300, 6600), z_stage_up=True, use_CNN_prediction_for_3d=True, downsample=None):
        # restore_path_camera_params: these will be used to update the camera parameters so we can generate rc_warp for
        # each camera accordingly
        # z_index_ref: None, an array, or 'full', where full means load whole stack
        # cam_slice0/1: defines which cameras to slice
        # z_index_ref should be of consistent shape wrt cam_slice0/1
        # grayscale: if True, debayer to grayscale; otherwise, debayer to RGB;
        # rc_global_shift: need to make sure the patches don't go beyond the edges of the gigamosaic; tune this manually
        # z_stage_up: whether the z-stack was taken with the stage moving up;
        # use_CNN_prediction_for_3d: if True, use CNN; if False, compute sharpness on the spot
        # downsample: integer for decimating data

        self.filepath = filepath
        self.restore_path_camera_params = restore_path_camera_params
        self.sensor_crop = sensor_crop
        self.camera_dims = camera_dims
        self.cam_slice0 = cam_slice0
        self.cam_slice1 = cam_slice1
        self.z_index_ref = z_index_ref
        self.grayscale = grayscale
        self.batch_size = batch_size
        self.rc_global_shift = np.array(rc_global_shift)
        self.z_stage_up = z_stage_up
        self.use_CNN_prediction_for_3d = use_CNN_prediction_for_3d
        self.downsample = downsample

        self.sensor_crop_base = self.sensor_crop  # sometimes we need to use the undownsampled version;
        if downsample is not None:
            self.sensor_crop = (self.sensor_crop[0] // downsample, self.sensor_crop[1] // downsample)
            self.rc_global_shift = (self.rc_global_shift[0] // downsample, self.rc_global_shift[1] // downsample)

    def load_camera_parameters(self, z_step_ratio_override=None):
        # load from restore_path_camera_params
        # z_step_ratio_override: if you want to override the one in restore_path_camera_params;

        restored = scipy.io.loadmat(self.restore_path_camera_params)
        # get precalibrated values:
        self.variable_initial_values_global = {key: restored[key].squeeze() for key in restored if '__' not in key}
        # get other settings:
        self.inds_keep = restored['inds_keep__'].flatten()
        self.recon_shape = restored['recon_shape__'].flatten()
        self.ul_offset = restored['ul_offset__'].flatten()
        self.array_dims = tuple(restored['camera_array_dims__'].flatten())  # cameras and lateral scans
        self.nominal_z_slices = restored['nominal_z_slices__'].flatten().astype(np.float32)[self.inds_keep]
        self.pre_downsample_factor = restored['pre_downsample_factor__'].flatten()
        if z_step_ratio_override is None:
            self.z_step_ratio = restored['z_step_ratio__'].squeeze()
        else:
            self.z_step_ratio = z_step_ratio_override
        # correct the nominal_z_slices:
        z_mean = self.nominal_z_slices.mean()
        self.nominal_z_slices = np.float32((self.nominal_z_slices - z_mean) * self.z_step_ratio + z_mean)

        self.gigamosaic_shape = (int(self.recon_shape[0] * self.pre_downsample_factor),
                                 int(self.recon_shape[1] * self.pre_downsample_factor))
        if self.downsample is not None:
            self.gigamosaic_shape = (self.gigamosaic_shape[0] // self.downsample,
                                     self.gigamosaic_shape[1] // self.downsample)

        assert len(self.inds_keep) == np.prod(self.array_dims)
        # doesn't handle the case when inds_keep is not the full array ...

        self.global_inds = np.arange(len(self.inds_keep)).reshape(self.array_dims)  # e.g., of shape (72, 48)
        self.xy_scans = (self.array_dims[0] // self.camera_dims[0],
                         self.array_dims[1] // self.camera_dims[1])  # dims of xy scan (e.g., 8, 8)

        # reshape camera parameters:
        V = self.variable_initial_values_global
        V['camera_height'] = V['camera_height'].reshape(self.array_dims)
        V['ground_surface_normal'] = V['ground_surface_normal'].reshape(self.array_dims + (3,))
        V['camera_in_plane_angle'] = V['camera_in_plane_angle'].reshape(self.array_dims)
        V['rc'] = V['rc'].reshape(self.array_dims + (2,)) * self.pre_downsample_factor
        V['illum_flat'] = V['illum_flat'].reshape(self.array_dims + (6,))  # may not use this ...
        self.camera_parameter_names = ['camera_height', 'ground_surface_normal', 'camera_in_plane_angle',
                                       'rc', 'illum_flat']  # note: this list only contains per-camera parameters
        # parameters that are shared among all parameters are handled separately (e.g., radial_camera_distortion)

        # also create zstitch object that can create the rc_warp parameters:
        self.rc_warper = zstitch(stack=np.zeros((1,) + self.sensor_crop_base + (1,)),
                                 ul_coords=np.zeros((1, 2)),  # (<^) these don't matter
                                 recon_shape=self.recon_shape, ul_offset=self.ul_offset,
                                 scale=1, batch_size=None, momentum=None, report_error_map=False, z_step_mm=1)  # dummy value for z_step_mm
        self.rc_warper.create_variables(deformation_model='camera_parameters',
                                        learning_rates={'camera_focal_length': -1e-3, 'camera_height': -1e-3,
                                                        'ground_surface_normal': -1e-3, 'camera_in_plane_angle': -1e-3,
                                                        'rc': -.1, 'gain': -1e-3, 'bias': -1e-3, 'illum_flat': -1e-3,
                                                        'radial_camera_distortion': -1e-3},
                                        variable_initial_values=None,  # will be adjusted later
                                        remove_global_transform=False, antialiasing_filter=False)
        self.camera_params_dict = {v.name[:-2]: v for v in self.rc_warper.train_var_list}  # to access variables easily
        # camera parameters shared by all cameras can be assigned now:
        self.camera_params_dict['radial_camera_distortion'].assign(V['radial_camera_distortion'][None, None])

        # this is what will be passed to warp_camera_parameters later in the generator:
        self.rc_warper.rc_base = np.transpose(self.rc_warper.rc_base,
                                              (0, 2, 1, 3)).reshape(self.rc_warper.num_images, -1, 2)
        self.rc_warper.rc_base = tf.cast(self.rc_warper.rc_base, tf.float32)

    def create_network(self, filters_list, skip_list, output_nonlinearity='linear', tf_ckpt_path=None):
        # create CNN and load parameters from checkpoint if desired
        self.network = fcnn(filters_list, skip_list, output_nonlinearity=output_nonlinearity, num_inputs=1)

        if tf_ckpt_path is not None:
            ckpt = tf.train.Checkpoint(network=self.network)
            manager = tf.train.CheckpointManager(ckpt, tf_ckpt_path, max_to_keep=2)
            ckpt.restore(manager.checkpoints[0])

    def create_tf_dataset(self, prefetch=1, requires_debayering=True):

        self.dataset = tf.data.Dataset.from_generator(lambda: self._get_mcam_generator(requires_debayering),
                                                      (tf.float32, tf.float32, tf.float32))
        self.dataset = self.dataset.batch(self.batch_size).prefetch(prefetch)

    def generate_gigamosaic(self, ignore_3d=False, margin=None, cnn_output_scale=1e-4, nc_save_path=None,
                            sigma=8, truncate=2, skip_blending=False):
        # While generate_full_recon operates on lateral scans from a single camera, this one generalizes to multiple
        # cameras. Because the reconstruction in this case is too large for the GPU, we do the CNN processing on the GPU
        # in the smallest bounding box reconstruction area, and then assign them to the gigapixel array on a CPU in
        # numpy.
        # ignore_3d: don't use CNN, and just use the homographic parameters; in this case, we still use CPU only because
        # the reconstruction may still be too large, even without the CNN.
        # margin: how many pixels along borders to crop out; only relevant if using CNN.
        # cnn_output_scale: this is the only parameter that's not saved when running optimization, so supply here.
        # nc_save_path and is only relevant when ignore_3d is True -- in addition to generating the gigamosaic, also
        # save .nc files of the all-in-focus images (rebayered) and height map (converted to uint8 as z-stack
        # indices, but not rebayered -- this means that the z-stack can't be bigger than 255 steps). If nc_save_path is
        # None, then don't save.
        # sigma and truncate are only relevant if not using CNN;
        # skip_blending: if True, then don't average in overlapped regions -- just replace existing pixels with new ones
        # but still create the normalize tensor to normalize out overlap occuring from pixels in the same image;

        sig_proj = .42465  # for interpolation

        if self.grayscale:
            num_gigamosaic_channels = 1
        else:
            num_gigamosaic_channels = 3
        if not ignore_3d:
            num_gigamosaic_channels += 1
            assert not self.grayscale  # for now, have to be RGB, because we use green channel for computing sharpness

            if not self.use_CNN_prediction_for_3d:
                num_gigamosaic_channels += 1  # also include a channel for sharpest sharpness

        gigamosaic = np.zeros(self.gigamosaic_shape + (num_gigamosaic_channels,), dtype=np.float32)
        normalize = np.zeros(self.gigamosaic_shape, dtype=np.float32)

        if not ignore_3d and self.use_CNN_prediction_for_3d:
            # create padding and depadding layers (this should only ever be used for full reconstruction generation):
            self.padded_shape = [self.network.get_compatible_size(dim) for dim in self.sensor_crop]
            pad_r = self.padded_shape[0] - self.sensor_crop[0]
            pad_c = self.padded_shape[1] - self.sensor_crop[1]
            pad_top = pad_r // 2
            pad_bottom = int(tf.math.ceil(pad_r / 2))
            pad_left = pad_c // 2
            pad_right = int(tf.math.ceil(pad_c / 2))
            pad_specs = ((pad_top, pad_bottom), (pad_left, pad_right))
            pad_layer = tf.keras.layers.ZeroPadding2D(pad_specs)
            depad_layer = tf.keras.layers.Cropping2D(pad_specs)

        if not ignore_3d:
            d0, d1 = np.indices(np.array(self.sensor_crop))  # for help with gathering below
            # to pass to self._compute_height_and_all_in_focus:
            self.d0 = d0
            self.d1 = d1

        if not ignore_3d and not self.use_CNN_prediction_for_3d:
            # gaussian blur kernel:
            w = 2 * int(truncate * sigma + 0.5) + 1
            x = np.arange(-w // 2, w // 2) + 1
            x, y = np.meshgrid(x, x)
            gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / 2 / sigma ** 2)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
            self.gaussian_kernel = gaussian_kernel.astype(np.float32)

            # derivative kernels:
            self.diff0_kernel = np.array([[1, -1]]).astype(np.float32)
            self.diff1_kernel = self.diff0_kernel.T

        total_iters = np.prod(self.xy_scans)  # start with number of patches per camera
        if self.cam_slice0 is None:
            total_iters *= self.camera_dims[0]
        else:
            total_iters *= int(np.diff(self.cam_slice0))
        if self.cam_slice1 is None:
            total_iters *= self.camera_dims[1]
        else:
            total_iters *= int(np.diff(self.cam_slice1))

        for im, rc_warp, nominal_z_slice in tqdm(self.dataset, total=total_iters):
            # if ignoring 3D, then im should be 1+1 or 3+1 channels, otherwise, im will have two channel dimensions, one
            # for color, one for z-stack

            if not ignore_3d and self.use_CNN_prediction_for_3d:
                CNN_input = im[:, :, :, :, 1]  # use the green channel; now it's 1, x, y, z

                CNN_input = tf.cast(CNN_input, dtype=tf.float32)  # cast from uint8 to float32; add batch dim


                # generate height map:
                im_pad = pad_layer(CNN_input)  # pad to a shape the network likes
                im = im[0]  # don't need batch dim anymore
                fcnn_out = self.network(im_pad)
                fcnn_depad = depad_layer(fcnn_out)[0]  # depad, and remove batch dimension

                depth = tf.reduce_mean(fcnn_depad,
                                       [-1]) * cnn_output_scale  # depth prediction; remove feature dimension

                # linearly interpolate depth value for indexing into stack
                # restrict to within range; casting is a flooring operation:
                num_z = tf.cast(tf.shape(im)[-2], dtype=tf.float32)
                depth_index_float = tf.math.sin(depth) * (num_z / 2 - .51) + num_z / 2 - .5
                depth_index_floor = tf.cast(depth_index_float, dtype=tf.int32)
                depth_index_ceil = depth_index_floor + 1
                dist2floor = depth_index_float - tf.cast(depth_index_floor, dtype=tf.float32)
                dist2ceil = 1 - dist2floor
                # ^these are all row x col

                im_all_in_focus_floor = tf.gather_nd(im, tf.stack([d0, d1, depth_index_floor], axis=2))
                im_all_in_focus_ceil = tf.gather_nd(im, tf.stack([d0, d1, depth_index_ceil], axis=2))
                # color channel should be carried over^

                im_all_in_focus_predict = (im_all_in_focus_floor * dist2ceil[:, :, None] +
                                           im_all_in_focus_ceil * dist2floor[:, :, None])

                # save to nc file:
                if nc_save_path is not None:
                    raise Exception('saving to nc file not yet implemented')

                # flatten out spatial dims (batch dim is 1):
                im_all_in_focus_predict = tf.reshape(im_all_in_focus_predict, [-1, num_gigamosaic_channels - 1])
                depth_index_float = tf.reshape(depth_index_float, [-1])
                if self.z_stage_up:
                    depth_index_float = depth_index_float - nominal_z_slice + num_z / 2  # sync to reference (flat target)
                else:
                    depth_index_float = depth_index_float + nominal_z_slice - num_z / 2  # sync to reference (flat target)

                # stacking:
                im = tf.concat([im_all_in_focus_predict, depth_index_float[:, None]], axis=1)  # add height as 4th channel

            elif not ignore_3d and not self.use_CNN_prediction_for_3d:
                # im is of shape z, x, y, color
                z_stack = im[0]  # remove batch dimension
                num_z = tf.cast(tf.shape(z_stack)[0], dtype=tf.float32)
                im = self._compute_height_and_all_in_focus(z_stack, nominal_z_slice - num_z / 2)
                # ^note that this function requires that the photometric stack be 3 channels, as the number of channels
                # is hard-coded in this function
            else:
                # all the channels you want should be in im already
                im = tf.reshape(im, [-1, num_gigamosaic_channels])

            recon_mini, normalize_mini, rc_min, rc_max = self._backproject_mini(rc_warp, im, sig_proj,
                                                                                num_gigamosaic_channels)

            # bring back to numpy and update gigamosaic:
            normalize_mini = normalize_mini.numpy()
            recon_mini = recon_mini.numpy()
            rc_min = rc_min.numpy()
            rc_max = rc_max.numpy()

            rc_min = rc_min - self.rc_global_shift
            rc_max = rc_max - self.rc_global_shift

            if margin is not None:  # needs to be done this late in the code because there can be black gaps in the mini
                # reconstructions due to rotation for example
                recon_mini = recon_mini[margin:-margin, margin:-margin]
                normalize_mini = normalize_mini[margin:-margin, margin:-margin]
                rc_min = rc_min + margin
                rc_max = rc_max - margin

            try:
                if skip_blending:
                    # just replace:
                    gigamosaic[rc_min[0]:rc_max[0], rc_min[1]:rc_max[1]] = recon_mini
                    normalize[rc_min[0]:rc_max[0], rc_min[1]:rc_max[1]] = normalize_mini
                else:
                    # accumulate:
                    gigamosaic[rc_min[0]:rc_max[0], rc_min[1]:rc_max[1]] += recon_mini
                    normalize[rc_min[0]:rc_max[0], rc_min[1]:rc_max[1]] += normalize_mini
            except:
                print('patch goes beyond gigamosaic borders, skipping ...')

        return gigamosaic, normalize

    @tf.function
    def _backproject_mini(self, rc_warp, im, sig_proj, num_gigamosaic_channels):
        # used by generate_gigamosaic; wrapped into a function to improve performance (avoids oom error for direct
        # sharpness calculations without CNN)

        # rc_warp doesn't need further warping
        rc_warp = tf.reshape(rc_warp, [-1, 2])
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

        rc_ff = rc_floor
        rc_cc = rc_ceil
        rc_cf = tf.stack([rc_ceil[:, 0], rc_floor[:, 1]], 1)
        rc_fc = tf.stack([rc_floor[:, 0], rc_ceil[:, 1]], 1)

        frc = tf.exp(-frc ** 2 / 2. / sig_proj ** 2)
        crc = tf.exp(-crc ** 2 / 2. / sig_proj ** 2)

        # augmented coordinates:
        rc_4 = tf.concat([rc_ff, rc_cc, rc_cf, rc_fc], 0)

        # interpolated:
        im_4 = tf.concat([im * frc[:, 0, None] * frc[:, 1, None],
                          im * crc[:, 0, None] * crc[:, 1, None],
                          im * crc[:, 0, None] * frc[:, 1, None],
                          im * frc[:, 0, None] * crc[:, 1, None]], 0)
        w_4 = tf.concat([frc[:, 0] * frc[:, 1],
                         crc[:, 0] * crc[:, 1],
                         crc[:, 0] * frc[:, 1],
                         frc[:, 0] * crc[:, 1]], 0)

        # get bounding box of the reconstruction, and keep track of coordinates:
        rc_min = tf.reduce_min(rc_4, axis=0)
        rc_max = tf.reduce_max(rc_4, axis=0)
        rc_4 = rc_4 - rc_min[None, :]
        mini_patch_shape = rc_max - rc_min

        # backproject:
        normalize_mini = tf.scatter_nd(rc_4, w_4, mini_patch_shape)
        recon_mini = tf.scatter_nd(rc_4, im_4, tf.concat([mini_patch_shape, [num_gigamosaic_channels]], axis=0))

        return recon_mini, normalize_mini, rc_min, rc_max

    @tf.function
    def _compute_height_and_all_in_focus(self, z_stack, nominal_z_slice, sharpness_channel=1):
        # z_stack is of shape num_z, x, y, color
        # sharpness channel: channel to compute the sharpness of
        # computes sharpest depth, all-in-focus RGB image, and sharpest sharpness

        # compute sharpness of z stack:
        z_stack_1_channel = z_stack[:, :, :, sharpness_channel:sharpness_channel + 1]
        gaussian_kernel = self.gaussian_kernel[:, :, None, None]
        diff0_kernel = self.diff0_kernel[:, :, None, None]
        diff1_kernel = self.diff1_kernel[:, :, None, None]
        blurred = tf.nn.conv2d(z_stack_1_channel, gaussian_kernel,
                               strides=1, padding='SAME')
        z_stack_hpf = z_stack_1_channel / blurred
        dx = tf.nn.conv2d(z_stack_hpf, diff0_kernel,
                          strides=1, padding='SAME')
        dy = tf.nn.conv2d(z_stack_hpf, diff1_kernel,
                          strides=1, padding='SAME')
        dxy = tf.sqrt(dx ** 2 + dy ** 2)
        dxy_blur = tf.nn.conv2d(dxy, gaussian_kernel,
                                strides=1, padding='SAME')[:, :, :, 0]

        # find depth:
        depth_index = tf.math.argmax(dxy_blur, axis=0)
        depth_index_float = tf.cast(depth_index, dtype=tf.float32)
        # and compute max:
        sharpest_sharpness = tf.reduce_max(dxy_blur, axis=0)  # turns out this is better than indexing using argmax
        # using depth index, compute all-in-focus image:
        all_in_focus = tf.gather_nd(z_stack, tf.stack([depth_index, self.d0, self.d1], axis=2))  # channels carry over

        if self.z_stage_up:
            depth_index_float = depth_index_float - nominal_z_slice
        else:
            depth_index_float = depth_index_float + nominal_z_slice

        # stack and reshape for backprojection:
        stacked = tf.concat([all_in_focus, depth_index_float[:, :, None], sharpest_sharpness[:, :, None]], axis=2)
        reshaped = tf.reshape(stacked, [-1, 5])  # 5: RGB, heightsharpest sharpness

        return reshaped

    def _get_mcam_generator(self, requires_debayering=True):
        # Adapted from mcam_loading_scripts.py's load_xyz function (basically copied load_xyz and edited in yield
        # statements). This generator will be converted into a tf dataset.
        # Instead of returning full datasets, this will yield individual images at a time; if only yielding one image
        # (and not a stack), then also yield z index.
        # If z_index_ref is not None, then use the full z-stack data and pick the z slice based on z_index_ref.
        # transpose_z_stack: whether to move the z-stack dimension away from the first dimension (for CNN); set to False
        # if computing sharpness so that the z-stack dimension can be treated as the batch dim for tf conv.

        if self.cam_slice0 is None:
            slice0 = slice(None)  # slice everything
        else:
            slice0 = slice(self.cam_slice0[0], self.cam_slice0[1])
        if self.cam_slice1 is None:
            slice1 = slice(None)
        else:
            slice1 = slice(self.cam_slice1[0], self.cam_slice1[1])

        if self.grayscale:
            cv_code = cv2.COLOR_BAYER_GB2GRAY
            num_channels = 1
        else:
            cv_code = cv2.COLOR_BAYER_GB2BGR
            num_channels = 3

        directories = list(d for d in Path(self.filepath).iterdir() if d.is_dir() and d.name[0] == 'y')
        directories = sorted(directories, key=filepath_key)

        y_name_initial = '_'.join(directories[0].name.split('_')[:2])
        x_name_initial = '_'.join(directories[0].name.split('_')[2:])
        y_steps = sum(d.name.startswith(y_name_initial) for d in directories)
        x_steps = sum(d.name.endswith(x_name_initial) for d in directories)
        directories = np.asarray(directories, dtype=object).reshape(y_steps, x_steps)
        one_dataset = xr.open_dataset(next(directories[0, 0].glob('best_images*')))
        bayer_pattern = str(one_dataset.bayer_pattern.data[0, 0])
        if 'mcam_data' in one_dataset:
            mcam_data = one_dataset.mcam_data
        elif 'images' in one_dataset:  # new version uses different names (also camera_x/y --> image_x/y)
            mcam_data = one_dataset.images
        else:
            raise Exception('invalid dataset')
        dtype = mcam_data.dtype
        dataset_original_shape = mcam_data.shape

        if self.cam_slice0 is None:  # take the original shape
            dim0 = dataset_original_shape[0]
            start0 = 0
            end0 = dataset_original_shape[0]
        else:  # shape defined by specified start and end
            dim0 = self.cam_slice0[1] - self.cam_slice0[0]
            start0 = self.cam_slice0[0]
            end0 = self.cam_slice0[1]
        if self.cam_slice1 is None:  # take the original shape
            dim1 = dataset_original_shape[1]
            start1 = 0
            end1 = dataset_original_shape[1]
        else:  # shape defined by specified start and end
            dim1 = self.cam_slice1[1] - self.cam_slice1[0]
            start1 = self.cam_slice1[0]
            end1 = self.cam_slice1[1]

        N_cameras_single = (dim0, dim1)
        N_cameras_total = (N_cameras_single[0] * y_steps, N_cameras_single[1] * x_steps)
        image_shape = dataset_original_shape[2:]
        dataset_expanded_shape = N_cameras_total + image_shape + (num_channels,)
        if type(self.z_index_ref) == np.ndarray:
            assert N_cameras_total == self.z_index_ref.shape  # make sure z_index_ref has the downsampled shape!

        # define some useful functions to be used below:
        def get_camera_params(i):
            # get camera parameters (for the current stage lateral position)
            # i is a 2-tuple index specifying camera
            camera_params = list()
            for var_name in self.camera_parameter_names:
                camera_params.append(
                    self.variable_initial_values_global[var_name][(i[0])::y_steps, (x_steps - i[1] - 1)::x_steps])
            return camera_params  # each element of this list has the first two dims corresponding to the slice0/slice1

        def get_rc_warp_params(camera_param_values):
            # get the homographic deformation coordinates
            # camera_param_values: a list of parameters with names given in self.camera_parameter_names, for a given
            # camera stack

            # change rc_warper's camera parameters:
            for var_name, camera_params_ in zip(self.camera_parameter_names, camera_param_values):
                self.camera_params_dict[var_name].assign(camera_params_[None])
            # generate rc_warp based on adjusted camera parameters:
            rc_warp_dense = self.rc_warper._warp_camera_parameters(self.rc_warper.rc_base,
                                                                   use_radial_deformation=False)
            rc_warp_dense = tf.reshape(rc_warp_dense,
                                       (self.rc_warper.num_images,
                                        self.rc_warper.stack.shape[1],
                                        self.rc_warper.stack.shape[2], 2))
            return rc_warp_dense

        if self.z_index_ref is None:
            for i in tqdm(np.ndindex(directories.shape), total=y_steps * x_steps, desc="Loading best data"):
                directory = directories[i]
                filename = next(directory.glob('best_images*'))
                dataset = xr.open_dataset(filename, engine='netcdf4').compute()
                z_pos = dataset.z_stage[slice0, slice1]
                # get z-stack index:
                if i == (0, 0):  # only need the first time
                    filename_z_stack = next(directory.glob('z_stack*'))
                    dataset_ = xr.open_dataset(filename_z_stack, engine='netcdf4').compute()
                    all_z_pos = dataset_.z_stage
                z_indices = (np.array(all_z_pos)[None, None, :] ==
                             np.array(z_pos)[:, :, None]).argmax(2)  # index of chosen z (for all cameras)

                # get camera parameters (for the current stage lateral position):
                camera_params = get_camera_params(i)

                # generator (is there a cleaner way of zipping iteration across multiple 2D arrays?):
                # (yield one camera at a time for this given lateral translation position)
                if 'mcam_data' in dataset:
                    mcam_data = dataset.mcam_data
                elif 'images' in dataset:  # new version uses different names (also camera_x/y --> image_x/y)
                    mcam_data = dataset.images
                else:
                    raise Exception('invalid dataset')
                for zipped in zip(mcam_data.data[slice0, slice1], z_indices, *camera_params):
                    for packed in zip(*zipped):
                        im = packed[0]
                        z_index = packed[1]
                        camera_params_unpacked = packed[2:]

                        if requires_debayering:
                            debayered = cv2.cvtColor(im, cv_code)
                        else:
                            debayered = im

                        if num_channels == 1:
                            debayered = debayered[..., None]

                        # change rc_warper's camera parameters and generate rc_warp based on adjusted camera parameters:
                        rc_warp_dense = get_rc_warp_params(camera_params_unpacked)

                        if self.downsample is None:
                            yield debayered, rc_warp_dense, z_index
                        else:
                            yield (debayered[::self.downsample, ::self.downsample],
                                   rc_warp_dense[:, ::self.downsample, ::self.downsample] / self.downsample, z_index)

        elif type(self.z_index_ref) != np.ndarray and self.z_index_ref == 'full':  # first conditional avoids warning print
            # load FULL z stack
            # this option doesn't need to yield z_index since you're yielding the whole debayered stack

            num_z = len(xr.open_dataset(next(directories[0, 0].glob('z_stack*'))).z_stage)
            nominal_z_slices_unflattened = self.nominal_z_slices.reshape(self.array_dims)
            # memory preallocations:
            preallocated_mcam_data = np.zeros((num_z, dim0, dim1, image_shape[0], image_shape[1]), dtype=dtype)
            debayered_z_stack = np.zeros((num_z,) + image_shape + (num_channels,), dtype=dtype)
            for i in tqdm(np.ndindex(directories.shape), total=y_steps * x_steps, desc="Loading full stack"):
                directory = directories[i]
                filename = next(directory.glob('z_stack*'))
                h5file = h5py.File(filename, 'r')
                keys = h5file.keys()
                if 'mcam_data' in keys:
                    dataset = h5file.get('mcam_data')
                elif 'images' in keys:
                    dataset = h5file.get('images')
                else:
                    raise Exception('invalid dataset')
                if self.cam_slice0 is None and self.cam_slice1 is None:
                    dataset.read_direct(preallocated_mcam_data)
                else:
                    the_slice = np.s_[:, start0:end0, start1:end1, :, :]
                    dataset.read_direct(preallocated_mcam_data, source_sel=the_slice)
                mcam_data = preallocated_mcam_data

                # get camera parameters (for the current stage lateral position):
                camera_params = get_camera_params(i)

                # generator; iterate by index, since mcam_data's first dimension is z-stack
                for r in range(start0, end0):
                    for c in range(start1, end1):
                        # get the camera parameters for the given lateral position FOV:
                        camera_params_unpacked = [params[r, c] for params in camera_params]

                        # change rc_warper's camera parameters and generate rc_warp based on adjusted camera parameters:
                        rc_warp_dense = get_rc_warp_params(camera_params_unpacked)

                        # debayer stack:
                        for z in range(num_z):
                            if requires_debayering:
                                debayered = cv2.cvtColor(mcam_data[z, r, c], cv_code)
                            else:
                                debayered = mcam_data[z, r, c]
                            if num_channels == 3:
                                debayered_z_stack[z, :, :, :] = debayered
                            elif num_channels == 1:
                                debayered_z_stack[z, :, :, 0] = debayered

                        if self.use_CNN_prediction_for_3d:
                            # move z dimension to second to last in preparation for CNN:
                            debayered_z_stack_T = np.transpose(debayered_z_stack, (1, 2, 0, 3))
                        else:
                            debayered_z_stack_T = debayered_z_stack

                        if self.downsample is None:
                            yield (debayered_z_stack_T, rc_warp_dense,
                                   nominal_z_slices_unflattened[r * self.xy_scans[0] + i[0],
                                                                (c + 1) * self.xy_scans[1] - i[1] - 1])
                            # NOTE! the scan direction is hard-coded -- that's why the second coordinate is flipped.
                            # Note that the only time we need to flip the direction to make it "wrong" is if we're
                            # looping through the file directory structure.
                        else:
                            assert not self.use_CNN_prediction_for_3d  # since the shape would be different
                            yield (debayered_z_stack_T[:, ::self.downsample, ::self.downsample],
                                   rc_warp_dense[:, ::self.downsample, ::self.downsample] / self.downsample,
                                   nominal_z_slices_unflattened[r * self.xy_scans[0] + i[0],
                                                                (c + 1) * self.xy_scans[1] - i[1] - 1])

        else:
            for i in tqdm(np.ndindex(directories.shape), total=y_steps * x_steps, desc="Loading selected data"):
                directory = directories[i]
                filename = next(directory.glob('z_stack*'))
                dataset = xr.open_dataset(filename, engine='netcdf4')

                # pick z slices according to z_index_ref:
                z_index_ref_subset = self.z_index_ref[(i[0])::y_steps, (x_steps - i[1] - 1)::x_steps]
                data_z_selected = np.empty(z_index_ref_subset.shape + image_shape + (num_channels,), dtype=dtype)
                for r in range(self.cam_slice0[0], self.cam_slice0[1]):
                    for c in range(self.cam_slice1[0], self.cam_slice1[1]):
                        if num_channels == 3:
                            data_z_selected[r - self.cam_slice0[0], c - self.cam_slice1[0]] = cv2.cvtColor(
                                np.asarray(dataset.mcam_data[z_index_ref_subset[r, c], r, c]), cv_code)
                        elif num_channels == 1:
                            data_z_selected[r - self.cam_slice0[0], c - self.cam_slice1[0], :, :, 0] = cv2.cvtColor(
                                np.asarray(dataset.mcam_data[z_index_ref_subset[r, c], r, c]), cv_code)

                data[(i[0])::y_steps, (x_steps - i[1] - 1)::x_steps] = data_z_selected
