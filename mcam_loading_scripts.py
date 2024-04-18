from pathlib import Path
import xarray as xr
import numpy as np
from tqdm.notebook import tqdm
import cv2


def load_xyz(filepath, z_index_ref=None, cam_slice0=None, cam_slice1=None, grayscale=False, keep_green_only=False):
    # for loading bga results, when data is organized bia directory; adapted from mark;
    # if z_index_ref is not None, then use the full z-stack data and pick the z slice based on z_index_ref
    # z_index_ref: None, an array, or 'full', where full means load whole stack
    # cam_slice0/1 defines which cameras to slice
    # z_index_ref should be of consistent shape wrt cam_slice0/1
    # grayscale: if True, debayer to grayscale; otherwise, debayer to RGB;
    # keep_green_only: only matters if not using grayscale, in which case only keep green channel. This saves
    # memory if loading full stack (and thus z_index_ref must be 'full')

    if cam_slice0 is None:
        slice0 = slice(None)  # slice everything
    else:
        slice0 = slice(cam_slice0[0], cam_slice0[1])
    if cam_slice1 is None:
        slice1 = slice(None)
    else:
        slice1 = slice(cam_slice1[0], cam_slice1[1])

    if grayscale:
        cv_code = cv2.COLOR_BAYER_GB2GRAY
        num_channels = 1
    else:
        cv_code = cv2.COLOR_BAYER_GB2BGR
        num_channels = 3

    if keep_green_only:
        # didn't test for the other modes
        assert z_index_ref == 'full'
        assert not grayscale
        num_channels = 1

    directories = list(d for d in Path(filepath).iterdir() if d.is_dir() and d.name[0] == 'y')
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

    if cam_slice0 is None:  # take the original shape
        dim0 = dataset_original_shape[0]
    else:  # shape defined by specified start and end
        dim0 = cam_slice0[1] - cam_slice0[0]
    if cam_slice1 is None:  # take the original shape
        dim1 = dataset_original_shape[1]
    else:  # shape defined by specified start and end
        dim1 = cam_slice1[1] - cam_slice1[0]

    N_cameras_single = (dim0, dim1)
    N_cameras_total = (N_cameras_single[0] * y_steps, N_cameras_single[1] * x_steps)
    image_shape = dataset_original_shape[2:]
    dataset_expanded_shape = N_cameras_total + image_shape + (num_channels,)
    if type(z_index_ref) == np.ndarray:
        assert N_cameras_total == z_index_ref.shape  # make sure z_index_ref has the downsampled shape!

    data = np.empty(dataset_expanded_shape, dtype=dtype)

    if z_index_ref is None:
        z_stage = np.empty(dataset_expanded_shape[:2], dtype=np.float32)
        z_index = np.empty(dataset_expanded_shape[:2], dtype=np.int32)
        for i in tqdm(np.ndindex(directories.shape), total=y_steps * x_steps, desc="Loading best data"):
            directory = directories[i]
            filename = next(directory.glob('best_images*'))
            dataset = xr.open_dataset(filename, engine='netcdf4').compute()
            if 'mcam_data' in dataset:
                mcam_data = dataset.mcam_data
            elif 'images' in dataset:  # new version uses different names (also camera_x/y --> image_x/y)
                mcam_data = dataset.images
            else:
                raise Exception('invalid dataset')
            debayered = np.stack([[cv2.cvtColor(im, cv_code) for im in im_]
                                  for im_ in mcam_data.data[slice0, slice1]])
            if num_channels == 1:
                debayered = debayered[..., None]
            data[(i[0])::y_steps, (x_steps - i[1] - 1)::x_steps] = debayered
            z_pos = dataset.z_stage[slice0, slice1]
            z_stage[(i[0])::y_steps, (x_steps - i[1] - 1)::x_steps] = z_pos

            # get z-stack index:
            if i == (0, 0):  # only need the first time
                filename_z_stack = next(directory.glob('z_stack*'))
                dataset = xr.open_dataset(filename_z_stack, engine='netcdf4').compute()
                all_z_pos = dataset.z_stage
            z_index[(i[0])::y_steps, (x_steps - i[1] - 1)::x_steps] = (np.array(all_z_pos)[None, None, :] ==
                                                                       np.array(z_pos)[:, :, None]).argmax \
                (2)  # index of chosen z
        return data, z_index

    elif type(z_index_ref) != np.ndarray and z_index_ref == 'full':  # first conditional avoids warning print
        # load FULL z stack
        num_z = len(xr.open_dataset(next(directories[0, 0].glob('z_stack*'))).z_stage)
        dataset_expanded_shape = (num_z,) + dataset_expanded_shape  # add z-stack dimension
        data = np.empty(dataset_expanded_shape, dtype=dtype)

        for i in tqdm(np.ndindex(directories.shape), total=y_steps * x_steps, desc="Loading full stack"):
            directory = directories[i]
            filename = next(directory.glob('z_stack*'))
            dataset = xr.open_dataset(filename, engine='netcdf4')  # .compute()

            if 'mcam_data' in dataset:
                mcam_data = dataset.mcam_data
            elif 'images' in dataset:  # new version uses different names (also camera_x/y --> image_x/y)
                mcam_data = dataset.images
            else:
                raise Exception('invalid dataset')
            subdata = np.empty((num_z,) + N_cameras_single + image_shape + (num_channels,), dtype=dtype)
            for r in range(cam_slice0[0], cam_slice0[1]):
                for c in range(cam_slice1[0], cam_slice1[1]):
                    for z in range(num_z):
                        if num_channels == 3:
                            # need to subtract out to rereference indices to 0:
                            subdata[z, r-cam_slice0[0], c-cam_slice1[0], :, :, :] = cv2.cvtColor(
                                np.asarray(mcam_data[z, r, c]), cv_code)
                        elif num_channels == 1:
                            if not keep_green_only:
                                subdata[z, r - cam_slice0[0], c - cam_slice1[0], :, :, 0] = cv2.cvtColor(
                                    np.asarray(mcam_data[z, r, c]), cv_code)
                            else:
                                subdata[z, r - cam_slice0[0], c - cam_slice1[0], :, :, 0] = cv2.cvtColor(
                                    np.asarray(mcam_data[z, r, c]), cv_code)[:, :, 1]

            data[:, (i[0])::y_steps, (x_steps - i[1] - 1)::x_steps] = subdata
        return data
    else:
        for i in tqdm(np.ndindex(directories.shape), total=y_steps * x_steps, desc="Loading selected data"):
            directory = directories[i]
            filename = next(directory.glob('z_stack*'))
            dataset = xr.open_dataset(filename, engine='netcdf4')  # .compute()
            if 'mcam_data' in dataset:
                mcam_data = dataset.mcam_data
            elif 'images' in dataset:  # new version uses different names (also camera_x/y --> image_x/y)
                mcam_data = dataset.images
            else:
                raise Exception('invalid dataset')
            # pick z slices according to z_index_ref:
            z_index_ref_subset = z_index_ref[(i[0])::y_steps, (x_steps - i[1] - 1)::x_steps]
            data_z_selected = np.empty(z_index_ref_subset.shape + image_shape + (num_channels,), dtype=dtype)
            for r in range(cam_slice0[0], cam_slice0[1]):
                for c in range(cam_slice1[0], cam_slice1[1]):
                    if num_channels == 3:
                        data_z_selected[r-cam_slice0[0], c-cam_slice1[0]] = cv2.cvtColor(
                            np.asarray(mcam_data[z_index_ref_subset[r, c], r, c]), cv_code)
                    elif num_channels == 1:
                        data_z_selected[r - cam_slice0[0], c - cam_slice1[0], :, :, 0] = cv2.cvtColor(
                            np.asarray(mcam_data[z_index_ref_subset[r, c], r, c]), cv_code)

            data[(i[0])::y_steps, (x_steps - i[1] - 1)::x_steps] = data_z_selected

        return data, z_index_ref


def filepath_key(filepath):
    parts = filepath.name.split('_')
    y = float(parts[1])
    x = float(parts[3])

    # should be a large number
    expected_steps_x = 1000

    return y * expected_steps_x + x

