"""
Created on Jan 11, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil

import numpy as np
import torch
import torch.nn.utils.rnn

import CAD120.metadata
import WNP.metadata


def collate_fn_cad(batch):
    features, labels, seg_lengths, total_length, activity, sequence_id = batch[0]
    feature_size = features[0].shape[1]
    label_num = len(CAD120.metadata.subactivities)

    max_seq_length = np.max(np.array([total_length for (features, labels, seg_lengths, total_length, activity, sequence_id) in batch]))
    features_batch = np.zeros((max_seq_length, len(batch), feature_size))
    labels_batch = np.ones((max_seq_length, len(batch))) * -1
    probs_batch = np.zeros((max_seq_length, len(batch), label_num))
    total_lengths = np.zeros(len(batch))
    ctc_labels = list()
    ctc_lengths = list()
    activities = list()
    sequence_ids = list()

    for batch_i, (features, labels, seg_lengths, total_length, activity, sequence_id) in enumerate(batch):
        current_len = 0
        ctc_labels.append(labels)
        ctc_lengths.append(len(labels))
        for seg_i, feature in enumerate(features):
            features_batch[current_len:current_len+seg_lengths[seg_i], batch_i, :] = np.repeat(features[seg_i], seg_lengths[seg_i], axis=0)
            labels_batch[current_len:current_len+seg_lengths[seg_i], batch_i] = labels[seg_i]
            probs_batch[current_len:current_len+seg_lengths[seg_i], batch_i, labels[seg_i]] = 1.0
            current_len += seg_lengths[seg_i]
        total_lengths[batch_i] = total_length
        activities.append(activity)
        sequence_ids.append(sequence_id)

    features_batch = torch.FloatTensor(features_batch)
    labels_batch = torch.LongTensor(labels_batch)
    probs_batch = torch.FloatTensor(probs_batch)
    total_lengths = torch.IntTensor(total_lengths)
    ctc_lengths = torch.IntTensor(ctc_lengths)

    return features_batch, labels_batch, probs_batch, total_lengths, ctc_labels, ctc_lengths, activities, sequence_ids


def collate_fn_wnp(batch):
    features, labels, seg_lengths, total_length, activity, sequence_id = batch[0]
    feature_size = features.shape[1]
    label_num = len(WNP.metadata.subactivities)

    max_seq_length = np.max(np.array([total_length for (features, labels, seg_lengths, total_length, activity, sequence_id) in batch]))
    features_batch = np.zeros((max_seq_length, len(batch), feature_size))
    labels_batch = np.ones((max_seq_length, len(batch))) * -1
    probs_batch = np.zeros((max_seq_length, len(batch), label_num))
    total_lengths = np.zeros(len(batch))
    ctc_labels = list()
    ctc_lengths = list()
    activities = list()
    sequence_ids = list()

    for batch_i, (features, labels, seg_lengths, total_length, activity, sequence_id) in enumerate(batch):
        features_batch[:total_length, batch_i, :] = np.nan_to_num(features)
        labels_batch[:total_length, batch_i] = labels
        for frame in range(features.shape[0]):
            probs_batch[frame, batch_i, labels[frame]] = 1.0

        merged_labels = list()
        current_label = -1
        for label in labels:
            if label != current_label:
                current_label = label
                merged_labels.append(current_label)
        ctc_labels.append(merged_labels)
        ctc_lengths.append(len(merged_labels))
        total_lengths[batch_i] = total_length
        activities.append(activity)
        sequence_ids.append(sequence_id)

    features_batch = torch.FloatTensor(features_batch)
    labels_batch = torch.LongTensor(labels_batch)
    probs_batch = torch.FloatTensor(probs_batch)
    total_lengths = torch.IntTensor(total_lengths)
    ctc_lengths = torch.IntTensor(ctc_lengths)

    return features_batch, labels_batch, probs_batch, total_lengths, ctc_labels, ctc_lengths, activities, sequence_ids


def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


def load_best_checkpoint(args, model, optimizer):
    # get the best checkpoint if available without training
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_epoch_error = checkpoint['best_epoch_error']
            try:
                avg_epoch_error = checkpoint['avg_epoch_error']
            except KeyError:
                avg_epoch_error = np.inf
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.cuda:
                model.cuda()
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
            return args, best_epoch_error, avg_epoch_error, model, optimizer
        else:
            print("=> no best model found at '{}'".format(best_model_file))
    return None


def main():
    pass


if __name__ == '__main__':
    main()
