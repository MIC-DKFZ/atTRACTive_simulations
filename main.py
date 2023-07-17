import matplotlib.pyplot as plt
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import random
import os
from main_utils import *


def main(subj_ids_test=None,
         bundles=None,
         n_cycles=20,
         num_prototypes=100,
         init_sample=20,
         samplesize=10,
         add_local_features=True,
         random_sampling=False,
         save=False,
         verbose=True,
         orig_data=False,
         path='data/'
         ):
    random.seed(42)
    np.random.seed(42)

    if random_sampling:
        random_sampling_string = 'Random'
    else:
        random_sampling_string = 'NoRandom'
    if add_local_features:
        add_local_features_string = 'AddLocalFeaures'
    else:
        add_local_features_string = 'NoAddLocalFeaures'

    if verbose:
        print('Random Sampling: ' + str(random_sampling))
        print('Adding Local Features: ' + str(add_local_features))

    for bundle in bundles:

        if verbose:
            print('Start Testing')

        for subj_id_test in subj_ids_test:
            if verbose:
                print('Subject: ' + subj_id_test)
                print('Loading Tractogram')

            # if original Tractogram is used downsample from 10 million to 1 million and correct the data
            # MitkDiffusion needs to be installed on the system for Groundtruth creation
            if orig_data:
                trac = load_trk('data/' + subj_id_test + '/All_10M.trk')

                # Downsample two 1 Mio Fibers
                trac = random.sample(trac, int(len(trac) * 0.1))

                # Resample to 40 points per fiber
                trac = set_number_of_points(trac, 40)

                # Save new tractogram
                create_trk(trac, 'data/' + subj_id_test + '/Trac.trk', save=True)

                # Reduce false-positives by first using the mask of the groundtruth and then start and end regions of tractseg
                os.system(
                    'MitkFiberExtractionRoi -i data/' + subj_id_test + '/Trac.trk -o data/' + subj_id_test + '/' + bundle + '.trk --rois data/' + subj_id_test + '/segmentations/' + bundle + '.nii.gz --overlap_fraction 1')
                os.system(
                    'MitkFiberExtractionRoi -i data/' + subj_id_test + '/' + bundle + '.trk -o data/' + subj_id_test + '/' + bundle + '.trk --rois data/' + subj_id_test + '/segmentations/' + bundle + '_merge.nii.gz --start_labels 1 --end_labels 2')
                os.system(
                    'MitkTractDensity -i  data/' + subj_id_test + '/' + bundle + '.trk -o data/' + subj_id_test + '/' + bundle + '_mask.nii.gz  --binary --reference_image data/MICCAI_Experiment/TractSeg/599671/ground_data/AF_left_mask.nii.gz')

            # Else use the already down- and resampled tractogram
            else:
                trac = load_trk('data/' + subj_id_test + '/Tractogram_preprocessed.trk')

            if verbose:
                print('Loading Bundle')
                print('Bundle: ', bundle)

            # Load groundtruth bundle to simulate the human expert.
            cur_bundle = load_trk('data/' + subj_id_test + '/' + bundle + '.trk')

            # Results directory for each bundle
            name = subj_id_test + '_' + bundle + '_' + random_sampling_string + '_' + add_local_features_string
            if verbose:
                print('Saving Results to: ' + name)

            if save:
                os.mkdir(path + name)
                os.mkdir(path + name + '/Predictions')
                os.mkdir(path + name + '/Entropyvectors')
                os.mkdir(path + name + '/Selection')
                os.mkdir(path + name + '/Pred_img')
                os.mkdir(path + name + '/Prototypes')

            if verbose:
                print('Building Reference vector of Groundtrouths (y_test)')

            # Find all streamlines in tractogram that appear in cur_bundle.
            # As the tractogram is downsampled, the groundtruth needs to be adjusted, too
            y_test = find_y_parallel2(set_number_of_points(trac, 4), set_number_of_points(cur_bundle, 4))

            if save:
                # Create trk from the new reduced groundtruth and the tractogram and save it
                create_trk(np.array(trac)[np.where(y_test > 0.5)[0]], path + name
                           + '/Pred_img/Groundtruth_bundle' + '.trk', save=True)
                create_trk(np.array(trac), path + name
                           + '/Pred_img/Trac_reduced_bundle' + '.trk', save=True)

            # Get prototypes with fft sampling (-init samples because these samples will also be added to the prototypes)
            prototypes = get_prototypes(trac, num_prototypes=num_prototypes - init_sample, fft=True)

            if verbose:
                print('Calculating Features')

            # Calculate Features based on d_mdf and d_end
            x_test = np.append(bundles_distances_mdf(trac, prototypes),
                               bundles_distances_mdf(set_number_of_points(trac, 2),
                                                     set_number_of_points(prototypes[:100], 2)),
                               axis=1)

            # Delete labeled idx from unlabeled_idx array
            if verbose:
                print('Get Train and Test Indices')
            test_idx = np.arange(len(trac))
            train_idx = np.array([])

            # Select 20 random streamlines for initialisation
            train_idx = np.append(train_idx, np.array(np.random.choice(test_idx, init_sample, replace=False))).astype(
                int)
            # Add one streamlines belonging to the target tract
            train_idx = np.append(train_idx, np.where(y_test == 1)[0][:1])

            # Add another streamline belonging two the target tract (which has the highest distance in feature space
            # to the first one)
            # Fist get features of target bundle
            cur_bundle = []
            for k in np.where(y_test == 1)[0]:
                cur_bundle.append(x_test[k])
            # Calculate distance in feature space from all streamlines of target bundle two first selected streamline in
            # the training data
            cur_line = []
            for k in range(len(cur_bundle)):
                cur_line.append(np.sqrt((np.square(x_test[train_idx[-1], :] - np.array(cur_bundle)[k, :]))).mean())
            # Add the other streamline two the training data
            train_idx = np.append(train_idx, np.where(y_test == 1)[0][int(np.argmax(cur_line))])
            # train_idx = np.append(train_idx, np.where(y_test == 1)[0][1:2])

            # Delete selected tracts from test data
            test_idx = test_idx[~np.isin(test_idx, train_idx)]

            if save:
                # Save the initial selected indices of streamliens
                np.save(path + name + '/Selection/selection_' + str(0) + '.npy',
                        train_idx)

            # Add local features
            if add_local_features:
                if verbose:
                    print('Adding local Features')
                if len(prototypes) < 200:
                    for k in train_idx:
                        prototypes.append(trac[k])

                if verbose:
                    print('Calculating Features')
                x_test = np.append(bundles_distances_mdf(trac, prototypes),
                                   bundles_distances_mdf(set_number_of_points(trac, 2),
                                                         set_number_of_points(prototypes[:100], 2)),
                                   axis=1)

            y_train = y_test[train_idx]
            x_train = x_test[train_idx, :]

            len_cycles = 0
            dice = []

            while len_cycles < n_cycles:
                if verbose:
                    print('Iteration: ' + str(len_cycles))

                x_train, y_train = shuffle(x_train, y_train)

                if verbose:
                    print('Training now with ' + str(x_train.shape[0]) + ' samples')

                if verbose:
                    print("Start training")
                # Start training random forest
                clf2 = RandomForestClassifier(n_jobs=-1, n_estimators=800, max_depth=50, class_weight='balanced',
                                              random_state=41)
                clf2.fit(x_train, y_train)

                if verbose:
                    print("Start predicting")
                # predict on all not yet annotated streamlines
                y_cur_pred = clf2.predict(x_test)
                y_pred_proba = clf2.predict_proba(x_test)

                if random_sampling:
                    selection = np.array(np.random.choice(test_idx, samplesize, replace=False))
                    train_idx = np.append(train_idx, selection).astype(int)

                else:
                    if verbose:
                        print("Get next Data")
                    selection, entropyvec = get_entropy(y_pred_proba, samplesize, train_idx)

                    train_idx = np.append(train_idx, selection).astype(int)

                if add_local_features:
                    if len(prototypes) < num_prototypes * 2:
                        if verbose:
                            print('Adding new local prototypes')
                        for k in selection:
                            prototypes.append(trac[k])

                        if verbose:
                            print('New size of Prototypes: ' + str(len(prototypes)))
                            print('Calculate and add new features')
                        x_test = np.append(bundles_distances_mdf(trac, prototypes),
                                           bundles_distances_mdf(set_number_of_points(trac, 2),
                                                                 set_number_of_points(prototypes[:100], 2)), axis=1)

                if save:
                    if verbose:
                        print('Saving Data of iteration')
                    np.save(path + name + '/Predictions/y_pred_' + str(len_cycles) + '.npy',
                            y_cur_pred)
                    if not random_sampling:
                        np.save(path + name + '/Entropyvectors/entropy_' + str(len_cycles) + '.npy',
                                entropyvec)
                    np.save(path + name + '/Selection/selection_' + str(len_cycles) + '.npy',
                            selection)
                    create_trk(np.append(np.array(trac)[test_idx][np.where(y_cur_pred[test_idx] > 0.5)[0]],
                                         np.array(trac)[train_idx][np.where(y_test[train_idx] > 0.5)[0]], axis=0),
                               path + name + '/Pred_img/Prediction_bundle_' +
                               str(len_cycles) + '.trk', save=True)

                    np.save(path + name + '/Prototypes/prototypes_' + str(len_cycles) + '.npy',
                            prototypes)

                dice.append(dsc(get_segmentation(path + name + '/Pred_img/Groundtruth_bundle' + '.trk').get_fdata(),
                                get_segmentation(path + name + '/Pred_img/Prediction_bundle_' +
                                                 str(len_cycles) + '.trk').get_fdata()))

                if verbose:
                    print('Dice is: ' + str(dice[-1]))

                if verbose:
                    print('Creating new Training Dataset')
                x_train = x_test[train_idx]
                y_train = y_test[train_idx]
                if verbose:
                    print('x_train now consists ' + str(len(train_idx)) + ' samples')

                test_idx = test_idx[~np.isin(test_idx, train_idx)]
                if verbose:
                    print('x_train now consists ' + str(len(test_idx)) + ' samples')

                len_cycles += 1

            if save:
                if verbose:
                    print('Saving Final Data')
                np.save(path + name + '/y_true.npy', y_test)
                np.save(path + name + '/train_idx.npy', train_idx)
                np.save(path + name + '/prototypes', prototypes)
                np.save(path + name + '/dice.npy', dice)

                fig2, ((ax2)) = plt.subplots(1, 1, figsize=(7, 4))  #
                ax2.plot(np.arange(22, 222, 10), dice)
                ax2.set_xticks(np.arange(25, 225, 25))
                ax2.set_ylim([-0.04, 1.04])
                plt.xlabel('Annotated streamlines')
                plt.ylabel('Dice')
                ax2.set_ylim([-0.04, 1.04])
                plt.title(bundle)
                plt.savefig(path + name + '/dice.png')
                plt.show()

    if verbose:
        print('Done with all')


subj_ids_test = ['645551']

path = 'data/'
if os.path.exists(path):
    bundles = ['OR_left', 'AF_left', 'CST_left']
    n_cycles = 20 # Number of active learning iterations
    samplesize = 10 # Number of streamlines to annotate each active learning iteration
    num_prototypes = 100 # Number of initial prototypes
    init_sample = 20 # Number of initial samples to annotate
    add_local_features = True # If true, adaptivee prototpyes are calculated
    random_sampling = False # If true, random sampling instead of entropy sampling
    orig_data = False # If true, data needs to be downsampled
    save = True
    verbose = True

    main(subj_ids_test=subj_ids_test,
         bundles=bundles,
         n_cycles=n_cycles,
         num_prototypes=num_prototypes,
         init_sample=init_sample,
         samplesize=samplesize,
         add_local_features=add_local_features,
         random_sampling=random_sampling,
         save=save,
         verbose=verbose,
         path=path
         )




