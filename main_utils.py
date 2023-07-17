import numpy as np
import nibabel as nib
from joblib import Parallel, delayed, cpu_count
from dipy.tracking.streamline import transform_streamlines

def load_trk(path):
    """
    Load a file as trk and convert it to an npy array.

    Parameters:
    path (str): The path to the trk file.

    list: A list of np.arrays representing the streamlines in the trk file.
    """

    print('Loading file as trk and converting it to an npy array')

    tract = nib.streamlines.load(path)
    tract_list = []

    for k in range(len(tract.streamlines)):
        tract_list.append(np.array(tract.streamlines[k], dtype=np.float32))

    return tract_list


def find_y_parallel2(tractogram, bundle):
    """
    Find labels in a tractogram for a given bundle using multithreading.

    Parameters:
    tractogram (list): A list of streamlines representing the tractogram.
    bundle (list): A list of streamlines representing the bundle to search for.
    save (bool, optional): Whether to save the results. Defaults to False.

    Returns:
    np.array: An array of labels indicating the presence (1) or absence (0) of the bundle in each streamline of the tractogram.
    """

    def my_y(t, y):
        # Find the position of y within t
        try:
            pos = np.where((t == y).all(2))[0][0]
        except IndexError:
            pos = None
        return pos

    bundle_array = np.array(bundle)
    tractogram_array = np.array(tractogram)

    n_jobs = cpu_count()
    print("Parallel computation of labels: %s cpus." % n_jobs)

    # Perform parallel computation of labels
    results = Parallel(n_jobs=n_jobs)(
        delayed(my_y)(tractogram_array, bundle_array[i]) for i in range(bundle_array.shape[0]))

    results = list(filter(None, results))

    print("Done.")

    y_list = np.zeros(len(tractogram), dtype=int)

    # Set labels to 1 where the bundle is present
    y_list[results] = 1

    return y_list


def get_prototypes(trac, num_prototypes=10, fft=True):
    num_data = int(np.ceil(3.0 * num_prototypes * np.log(num_prototypes)))
    subset = np.array(trac[:num_data])
    if fft:
        index = np.arange(num_data)
        distancematrix = []
        for i in range(len(subset)):
            cur_line = []
            for k in range(len(subset)):
                cur_line.append(np.sqrt((np.square(subset[i, :] - subset[k, :]))).mean())
            distancematrix.append(cur_line)
        distancematrix = np.array(distancematrix)
        # / *Index to find values in distancematrix * /
        myidx = []
        # / *Index to find actual streamlines using indexUnc * /
        indexUncDist = []
        # / *Start  with the Streamline of the highest entropy, which is in distance_matrix at idx 0 * /
        myidx.append(0);
        indexUncDist.append(index[myidx[0]])

        # / *Vecotr that stores minvalues of current iteration * /

        sum_matrix = np.empty((0, len(subset)))
        for i in range(num_prototypes - 1):
            cur_vec = []
            # / *Save mean distance of all used Samples * /

            sum_matrix = np.append(sum_matrix, np.expand_dims(distancematrix[myidx[-1]], 0), axis=0)

            for k in range(sum_matrix.shape[1]):
                cur_vec.append(sum_matrix[:, k].min())
            myidx.append(np.argmax(cur_vec))
            indexUncDist.append(index[myidx[i + 1]])
        prototypes = []
        for k in myidx:
            prototypes.append(trac[k])

        return prototypes
    else:
        return subset


def get_entropy(votes, samplesize, train_idx):
    """
    Compute the entropy vector based on the votes and select samples based on entropy.

    Parameters:
    votes (np.array): The votes array.
    samplesize (int): The number of samples to select.
    train_idx (list): The indices of training samples.

    Returns:
    tuple: A tuple containing the selected indices and the entropy vector.
    """

    entropy_vec = []

    # Compute the entropy for each vote category
    for k in range(0, votes.shape[0]):
        entropy_vec.append(
            - (votes[k][0] / (votes[k][0] + votes[k][1])) * np.log2(votes[k][0] / (votes[k][0] + votes[k][1])) -
            (votes[k][1] / (votes[k][0] + votes[k][1])) * np.log2(votes[k][1] / (votes[k][0] + votes[k][1])))

    # Sort the indices based on entropy in descending order
    selection = np.flip(np.argsort(np.nan_to_num(entropy_vec)))

    # Remove already selected training indices from the selection
    selection = selection[~np.isin(selection, train_idx)]


    return selection[:samplesize], np.nan_to_num(entropy_vec)


def dsc(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0:
        return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

def create_trk(streamlines, out_file, affine=np.zeros((4, 4)), vox_sizes=np.array([0, 0, 0]), vox_order='RAS',
               dim=np.array([0, 0, 0]), save=False):
    """
    The default values for the parameters are the values for the HCP data.
    """
    if affine.any() == 0:
        affine = np.array([[1.25, 0., 0., 90.],
                           [0., 1.25, 0., -126.],
                           [0., 0., 1.25, -72.],
                           [0., 0., 0., 1.]],
                          dtype=np.float32)
    if (vox_sizes == [0, 0, 0]).all():
        vox_sizes = np.array([1.25, 1.25, 1.25], dtype=np.float32)
    if (dim == [0, 0, 0]).all():
        dim = np.array([145, 174, 145], dtype=np.int16)
    if out_file.split('.')[-1] != 'trk':
        print("Format not supported.")

    # Create a new header with the correct affine
    hdr = nib.streamlines.trk.TrkFile.create_empty_header()
    hdr['voxel_sizes'] = vox_sizes
    hdr['voxel_order'] = vox_order
    hdr['dimensions'] = dim
    hdr['voxel_to_rasmm'] = affine
    hdr['nb_streamlines'] = len(streamlines)
    if not len(streamlines)==0:
        if streamlines[0].shape[0] == 0:
            streamlines = streamlines[1:]
    t = nib.streamlines.tractogram.Tractogram(streamlines=streamlines, affine_to_rasmm=np.eye(4))
    if save:
        nib.streamlines.save(t, out_file, header=hdr)

    return t


def get_segmentation(img_path, save=False):
    # Load the streamline bundle from the TRK file
    streams = nib.streamlines.load(img_path)
    # Convert the streamlines using the affine matrix
    affine = streams.header['voxel_to_rasmm']
    transformed_streamlines = transform_streamlines(streams.streamlines, np.linalg.inv(affine))
    # Get the voxel dimensions from the header
    voxel_size = streams.header['voxel_sizes']
    voxel_dimensions = streams.header['dimensions']
    # Create an empty binary mask volume
    mask = np.zeros(voxel_dimensions)

    # Mark the voxels traversed by the streamlines as 1 in the mask
    for streamline in transformed_streamlines:
        indices = np.round(streamline).astype(int)
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    # Save the segmentation mask as an image file
    mask_img = nib.Nifti1Image(mask.astype(np.float32), affine)
    if save:
        mask_path = input[:-4] +'_mask.nii.gz'
        nib.save(mask_img, mask_path)

    return mask_img


