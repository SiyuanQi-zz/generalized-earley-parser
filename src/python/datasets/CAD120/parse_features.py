"""
Created on Jan 11, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import pickle

import numpy as np

import metadata
import config


def parse_colon_seperated_features(colon_seperated):
    f_list = [int(x.split(':')[1]) for x in colon_seperated]
    return f_list


def read_features(segments_feature_path, filename):
    filename_base = os.path.basename(filename)
    sequence_id = filename_base.split('_')[0]
    segment_index = int(os.path.splitext(filename_base)[0].split('_')[1])

    # Spatial features
    with open(filename) as f:
        first_line = f.readline().strip()
        object_num = int(first_line.split(' ')[0])
        object_object_num = int(first_line.split(' ')[1])
        skeleton_object_num = int(first_line.split(' ')[2])

        # 0:160 temporal features 160, skeleton features 630, skeleton-object features 400
        human_features = np.zeros((1, 1200))

        # Object feature
        for _ in range(object_num):
            line = f.readline()

        # Skeleton feature
        line = f.readline()
        colon_seperated = [x.strip() for x in line.strip().split(' ')]
        gt_label = int(colon_seperated[0]) - 1
        human_features[0, 165:795] = parse_colon_seperated_features(colon_seperated[2:])

        # Object-object feature
        for _ in range(object_object_num):
            line = f.readline()

        # Skeleton-object feature
        skeleton_object_feature = np.zeros((1, 400))
        for _ in range(skeleton_object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            s_o_id = int(colon_seperated[2])
            skeleton_object_feature += np.array(parse_colon_seperated_features(colon_seperated[3:]))
        skeleton_object_feature /= skeleton_object_num
        human_features[0, 800:] = skeleton_object_feature

    # Temporal features
    if segment_index == 1:
        pass
    else:
        with open(os.path.join(segments_feature_path, '{}_{}_{}.txt'.format(sequence_id, segment_index-1, segment_index)), 'r') as f:
            first_line = f.readline().strip()
            object_object_num = int(first_line.split(' ')[0])
            skeleton_skeleton_num = int(first_line.split(' ')[1])
            assert skeleton_skeleton_num == 1

            # Object-object temporal feature
            for _ in range(object_object_num):
                line = f.readline()

            # Skeleton-object temporal feature
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            human_features[0, 0:160] = parse_colon_seperated_features(colon_seperated[3:])

    return human_features, gt_label


def expand_data(paths, features, gt_labels):
    # Read video segmentations
    training_data = dict()
    testing_data = dict()
    last_sequence_id = ''
    for datadir in os.listdir(os.path.join(paths.cad_data_root)):
        datadir = os.path.join(paths.cad_data_root, datadir)
        if os.path.isdir(datadir) and datadir.endswith('annotations'):
            subject = os.path.split(datadir)[1].strip('_annotations')
            data = dict()
            for event in os.listdir(datadir):
                eventdir = os.path.join(datadir, event)

                # Parse data into spatial-temporal parse graphs
                with open(os.path.join(eventdir, 'labeling.txt')) as f:
                    i_line = 0
                    for line in f:
                        sequence_labeling = line.strip().split(',')
                        sequence_id = sequence_labeling[0]
                        if sequence_id != last_sequence_id:
                            data[sequence_id] = dict()
                            data[sequence_id]['activity'] = event
                            data[sequence_id]['features'] = list()
                            data[sequence_id]['labels'] = list()
                            data[sequence_id]['seg_lengths'] = list()
                            data[sequence_id]['total_length'] = 0
                            # data[sequence_id]['features'] = np.zeros((0, 1200))
                            # data[sequence_id]['labels'] = list()
                            last_sequence_id = sequence_id
                            i_line = 0
                        # print i_line, sequence_labeling
                        start_frame = int(sequence_labeling[1])
                        end_frame = int(sequence_labeling[2])
                        frame_num = end_frame - start_frame + 1
                        subactivity = sequence_labeling[3]
                        affordance_labels = sequence_labeling[4:]
                        if i_line == len(gt_labels[sequence_id]):
                            continue
                        assert metadata.subactivities[gt_labels[sequence_id][i_line]] == subactivity

                        data[sequence_id]['features'].append(features[sequence_id][i_line])
                        data[sequence_id]['labels'].append(gt_labels[sequence_id][i_line])
                        data[sequence_id]['seg_lengths'].append(frame_num)
                        data[sequence_id]['total_length'] += frame_num
                        # data[sequence_id]['features'] = np.vstack((data[sequence_id]['features'], np.repeat(features[sequence_id][i_line], frame_num, axis=0)))
                        # data[sequence_id]['labels'].extend([gt_labels[sequence_id][i_line] for _ in range(frame_num)])

                        i_line += 1

            if subject != 'Subject5':
                training_data.update(data)
            else:
                testing_data.update(data)

    datapath = os.path.join(paths.tmp_root, 'data')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    pickle.dump(training_data, open(os.path.join(datapath, 'cad_training.p'), 'wb'))
    pickle.dump(testing_data, open(os.path.join(datapath, 'cad_testing.p'), 'wb'))


def collect_data(paths):
    if not os.path.exists(paths.tmp_root):
        os.makedirs(paths.tmp_root)
    segments_files_path = os.path.join(paths.cad_data_root, 'features_cad120_ground_truth_segmentation', 'segments_svm_format')
    segments_feature_path = os.path.join(paths.cad_data_root, 'features_cad120_ground_truth_segmentation', 'features_binary_svm_format')

    # Read features and ground truth labels
    features = dict()
    gt_labels = dict()
    sequence_ids = list()
    for sequence_path_file in os.listdir(segments_files_path):
        sequence_id = os.path.splitext(sequence_path_file)[0]
        features[sequence_id] = list()
        gt_labels[sequence_id] = list()
        sequence_ids.append(sequence_id)

        with open(os.path.join(segments_files_path, sequence_path_file)) as f:
            first_line = f.readline()
            segment_feature_num = int(first_line.split(' ')[0])

            for _ in range(segment_feature_num):
                segment_feature_filename = f.readline().strip()
                human_features, gt_label = read_features(segments_feature_path, os.path.join(segments_feature_path, os.path.basename(segment_feature_filename)))
                features[sequence_id].append(human_features)
                gt_labels[sequence_id].append(gt_label)

    expand_data(paths, features, gt_labels)


def main():
    paths = config.Paths()
    start_time = time.time()
    collect_data(paths)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
