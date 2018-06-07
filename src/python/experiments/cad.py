"""
Created on Jan 11, 2018

@author: Siyuan Qi

Description of the file.

"""

# System imports
import os
import time
import datetime
import argparse
import decimal
import json

# Libraries
import numpy as np
import torch
import torch.autograd
import sklearn.metrics
import warpctc_pytorch

# Local imports
import config
import logutil
import utils
import models
import datasets
import vizutil
import parser.grammarutils

cross_entropy = torch.nn.CrossEntropyLoss().cuda()
mse_loss = torch.nn.MSELoss().cuda()
softmax = torch.nn.Softmax(dim=2)
ctc_loss = warpctc_pytorch.CTCLoss().cuda()


def loss_func(model_outputs, labels, probs, ctc_labels, total_lengths, ctc_lengths):
    loss = 0
    for i_batch in range(model_outputs.size()[1]):
        seg_length = int(total_lengths[i_batch])
        # loss += cross_entropy(model_outputs[:seg_length, i_batch, :], labels[:seg_length, i_batch])
        loss += mse_loss(model_outputs[:seg_length, i_batch, :], probs[:seg_length, i_batch, :])
    # loss = loss + ctc_loss(softmax(model_outputs), ctc_labels, total_lengths, ctc_lengths).cuda()/torch.sum(total_lengths).data[0]
    return loss


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))

    # Load data
    training_set, testing_set, train_loader, test_loader = utils.get_cad_data(args)
    features, labels, seg_lengths, total_length, activity, sequence_id = training_set[0]
    feature_size = features[0].shape[1]
    label_num = len(datasets.cad_metadata.subactivities)
    hidden_size = 256
    hidden_layers = 2

    model = models.BLSTM(feature_size, hidden_size, hidden_layers, label_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = loss_func
    if args.cuda:
        model = model.cuda()

    loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    epoch_errors = list()
    avg_epoch_error = np.inf
    best_epoch_error = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        logger.log_value('learning_rate', args.lr).step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger, args=args)
        # test on validation set
        if epoch > 10:
            epoch_error = validate(test_loader, model, args=args)
        else:
            epoch_error = 1.0

        epoch_errors.append(epoch_error)
        if len(epoch_errors) == 10:
            new_avg_epoch_error = np.mean(np.array(epoch_errors))
            if avg_epoch_error - new_avg_epoch_error < 0.01:
                args.lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            avg_epoch_error = new_avg_epoch_error
            epoch_errors = list()

        is_best = epoch_error < best_epoch_error
        best_epoch_error = min(epoch_error, best_epoch_error)
        datasets.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                        'best_epoch_error': best_epoch_error, 'avg_epoch_error': avg_epoch_error,
                                        'optimizer': optimizer.state_dict(), },
                                       is_best=is_best, directory=args.resume)
        print('best_epoch_error: {}, avg_epoch_error: {}'.format(best_epoch_error,  avg_epoch_error))

    # For testing
    loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint
    validate(test_loader, model, args=args, test=True)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def train(train_loader, model, criterion, optimizer, epoch, logger, args=None):
    batch_time = logutil.AverageMeter()
    data_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()
    subactivity_error_ratio = logutil.AverageMeter()

    # switch to train mode
    model.train()

    end_time = time.time()
    for i, (features, labels, probs, total_lengths, ctc_labels, ctc_lengths, activities, sequence_ids) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        features = utils.to_variable(features, args.cuda)
        labels = utils.to_variable(labels, args.cuda)
        probs = utils.to_variable(probs, args.cuda)

        total_lengths = torch.autograd.Variable(total_lengths)
        ctc_labels = torch.autograd.Variable(torch.IntTensor([item for sublist in ctc_labels for item in sublist]))
        ctc_lengths = torch.autograd.Variable(ctc_lengths)

        model_outputs = model(features)
        _, pred_labels = torch.max(model_outputs, dim=2)
        train_loss = criterion(model_outputs, labels, probs, ctc_labels, total_lengths, ctc_lengths)

        # Log
        losses.update(train_loss.data[0], torch.sum(total_lengths).data[0])

        subact_micro_result = sklearn.metrics.precision_recall_fscore_support(labels.cpu().data.numpy().flatten().tolist(), pred_labels.cpu().data.numpy().flatten().tolist(), labels=range(10), average='micro')
        subactivity_error_ratio.update(1.0-subact_micro_result[0], torch.sum(total_lengths).data[0])

        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)

    print('Epoch: [{0}] Avg Subactivity Error Ratio {act_err.avg:.3f}; Average Loss {losses.avg:.3f}; Batch Avg Time {b_time.avg:.3f}'
          .format(epoch, act_err=subactivity_error_ratio, losses=losses, b_time=batch_time))


def inference(model_outputs, activities, sequence_ids, ctc_labels, args):
    model_output_probs = softmax(model_outputs)
    model_output_probs = model_output_probs.data.cpu().numpy()
    batch_earley_pred_labels = list()
    batch_tokens = list()
    batch_seg_pos = list()
    for batch_i in range(model_outputs.size()[1]):
        grammar_file = os.path.join(args.tmp_root, 'grammar', 'cad', activities[batch_i]+'.pcfg')
        grammar = parser.grammarutils.read_grammar(grammar_file, index=True, mapping=datasets.cad_metadata.subactivity_index)
        gen_earley_parser = parser.GeneralizedEarley(grammar)
        best_string, prob = gen_earley_parser.parse(np.squeeze(model_output_probs[:, batch_i, :]))
        # print activities[batch_i], sequence_ids[batch_i], model_output_probs.shape[0]
        # print ctc_labels[batch_i]
        # print [int(s) for s in best_string.split()], "{:.2e}".format(decimal.Decimal(prob))

        # Back trace to get labels of the entire sequence
        earley_pred_labels, tokens, seg_pos = gen_earley_parser.compute_labels()
        batch_earley_pred_labels.append(earley_pred_labels)
        batch_tokens.append(tokens)
        batch_seg_pos.append(seg_pos)

    _, nn_pred_labels = torch.max(model_outputs, dim=2)

    return nn_pred_labels, batch_earley_pred_labels, batch_tokens, batch_seg_pos


def predict(activities, total_lengths, labels, ctc_labels, batch_tokens, batch_seg_pos):
    np_labels = labels.data.cpu().numpy().astype(np.int)

    # Ground truth segmentation
    gt_batch_seg_pos = list()
    for batch_i in range(np_labels.shape[1]):
        gt_batch_seg_pos.append(list())
        token_i = 0
        for frame in range(total_lengths[batch_i]):
            if int(np_labels[frame, batch_i]) != int(ctc_labels[batch_i][token_i]):
                gt_batch_seg_pos[batch_i].append(frame - 1)
                token_i += 1
        gt_batch_seg_pos[batch_i].append(int(total_lengths[batch_i]) - 1)

    with open(os.path.join(args.tmp_root, 'prior', 'cad', 'duration_prior.json')) as f:
        duration_prior = json.load(f)

    # Segment and frame-wise prediction
    duration = 45
    gt_seg_predictions = list()
    gt_frame_predictions = list()
    seg_predictions = list()
    frame_predictions = list()
    for batch_i in range(np_labels.shape[1]):
        grammar_file = os.path.join(args.tmp_root, 'grammar', 'cad', activities[batch_i]+'.pcfg')
        grammar = parser.grammarutils.read_grammar(grammar_file, index=True, mapping=datasets.cad_metadata.subactivity_index)
        token_i = -1
        gt_token_i = -1
        for frame in range(total_lengths[batch_i]-duration):
            # Segment prediction
            if gt_token_i == -1 or gt_batch_seg_pos[batch_i][gt_token_i] < frame:
                gt_token_i += 1
                if gt_token_i < len(ctc_labels[batch_i])-1:
                    current_gt_prediction = ctc_labels[batch_i][gt_token_i+1]
                else:
                    current_gt_prediction = ctc_labels[batch_i][-1]
            gt_seg_predictions.append(current_gt_prediction)
            gt_frame_predictions.extend(np_labels[frame:frame+duration, batch_i].tolist())

            if token_i == -1 or batch_seg_pos[batch_i][token_i] < frame:
                token_i += 1
                current_token = batch_tokens[batch_i][token_i]
                current_tokens = batch_tokens[batch_i][:token_i+1]
                symbols, probs = parser.grammarutils.earley_predict(grammar, current_tokens)
                if probs:
                    current_prediction = int(symbols[np.argmax(probs)])
                else:
                    current_prediction = int(batch_tokens[batch_i][-1])

                # Predict current segment length
                mu, sigma = duration_prior[current_token]
                current_token = int(current_token)
                sample_duration = int(mu)
                seg_start_frame = frame

            seg_predictions.append(current_prediction)
            duration_prediction = [current_token for _ in range(min(duration, max(30, sample_duration-(frame-seg_start_frame))))]
            duration_prediction.extend([current_prediction for _ in range(duration-len(duration_prediction))])
            frame_predictions.extend(duration_prediction)

    return gt_seg_predictions, gt_frame_predictions, seg_predictions, frame_predictions


def validate(val_loader, model, args, test=False):
    def compute_accuracy(gt_results, results, metric='micro'):
        return sklearn.metrics.precision_recall_fscore_support(gt_results, results, labels=range(10), average=metric)

    batch_time = logutil.AverageMeter()
    baseline_acc_ratio = logutil.AverageMeter()
    subactivity_acc_ratio = logutil.AverageMeter()
    seg_pred_acc_ratio = logutil.AverageMeter()
    frame_pred_acc_ratio = logutil.AverageMeter()

    all_baseline_detections = list()
    all_gt_detections = list()
    all_detections = list()
    all_gt_seg_predictions = list()
    all_gt_frame_predictions = list()
    all_seg_predictions = list()
    all_frame_predictions = list()

    # switch to evaluate mode
    model.eval()

    end_time = time.time()
    for i, (features, labels, probs, total_lengths, ctc_labels, ctc_lengths, activities, sequence_ids) in enumerate(val_loader):
        features = utils.to_variable(features, args.cuda)
        labels = utils.to_variable(labels, args.cuda)

        total_lengths = torch.autograd.Variable(total_lengths)

        # Inference
        model_outputs = model(features)
        pred_labels, batch_earley_pred_labels, batch_tokens, batch_seg_pos = inference(model_outputs, activities, sequence_ids, ctc_labels, args)

        # Visualize results
        for batch_i in range(labels.size()[1]):
            vizutil.plot_segmentation(
                [labels[:, batch_i].squeeze(), pred_labels[:, batch_i].squeeze(), batch_earley_pred_labels[batch_i]],
                int(total_lengths[batch_i]), filename=os.path.join(args.tmp_root, 'visualize', 'segmentation', 'cad', '{}_{}.pdf'.format(activities[batch_i], sequence_ids[batch_i])), border=False, vmax=len(datasets.cad_metadata.subactivities))

        # Evaluation
        # Frame-wise detection
        baseline_detections = pred_labels.cpu().data.numpy().flatten().tolist()
        gt_detections = labels.cpu().data.numpy().flatten().tolist()
        detections = [l for pred_labels in batch_earley_pred_labels for l in pred_labels.tolist()]
        all_baseline_detections.extend(baseline_detections)
        all_gt_detections.extend(gt_detections)
        all_detections.extend(detections)
        baseline_micro_result = compute_accuracy(gt_detections, baseline_detections)
        subact_micro_result = compute_accuracy(gt_detections, detections)

        gt_seg_predictions, gt_frame_predictions, seg_predictions, frame_predictions = predict(activities, total_lengths, labels, ctc_labels, batch_tokens, batch_seg_pos)
        all_gt_seg_predictions.extend(gt_seg_predictions)
        all_gt_frame_predictions.extend(gt_frame_predictions)
        all_seg_predictions.extend(seg_predictions)
        all_frame_predictions.extend(frame_predictions)
        seg_pred_result = compute_accuracy(gt_seg_predictions, seg_predictions)
        frame_pred_result = compute_accuracy(gt_frame_predictions, frame_predictions)

        baseline_acc_ratio.update(baseline_micro_result[0], torch.sum(total_lengths).data[0])
        subactivity_acc_ratio.update(subact_micro_result[0], torch.sum(total_lengths).data[0])
        seg_pred_acc_ratio.update(seg_pred_result[0], torch.sum(total_lengths).data[0])
        frame_pred_acc_ratio.update(frame_pred_result[0], len(all_gt_frame_predictions))

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    print(' * Baseline Accuracy Ratio {base_acc.avg:.3f}; '.format(base_acc=baseline_acc_ratio))
    print(' * Detection Accuracy Ratio {act_acc.avg:.3f}; Segment Prediction Accuracy Ratio Batch Avg {seg_pred_acc.avg:.3f}; Frame Prediction Accuracy Ratio Batch Avg {frame_pred_acc.avg:.3f}; Time {b_time.avg:.3f}'
          .format(act_acc=subactivity_acc_ratio, seg_pred_acc=seg_pred_acc_ratio, frame_pred_acc=frame_pred_acc_ratio, b_time=batch_time))
    print(compute_accuracy(all_gt_detections, all_baseline_detections, metric='macro'))
    print(compute_accuracy(all_gt_detections, all_detections, metric='macro'))
    print(compute_accuracy(all_gt_seg_predictions, all_seg_predictions, metric='macro'))
    print(compute_accuracy(all_gt_frame_predictions, all_frame_predictions, metric='macro'))

    confusion_matrix = sklearn.metrics.confusion_matrix(all_gt_detections, all_detections, labels=range(len(datasets.cad_metadata.subactivities)))
    vizutil.plot_confusion_matrix(confusion_matrix, datasets.cad_metadata.subactivities[:], normalize=True, title='', filename=os.path.join(args.tmp_root, 'visualize', 'confusion', 'cad', 'detection.pdf'))
    confusion_matrix = sklearn.metrics.confusion_matrix(all_gt_frame_predictions, all_frame_predictions, labels=range(len(datasets.cad_metadata.subactivities)))
    vizutil.plot_confusion_matrix(confusion_matrix, datasets.cad_metadata.subactivities[:], normalize=True, title='', filename=os.path.join(args.tmp_root, 'visualize', 'confusion', 'cad', 'prediction_frame.pdf'))
    confusion_matrix = sklearn.metrics.confusion_matrix(all_gt_seg_predictions, all_seg_predictions, labels=range(len(datasets.cad_metadata.subactivities)))
    vizutil.plot_confusion_matrix(confusion_matrix, datasets.cad_metadata.subactivities[:], normalize=True, title='', filename=os.path.join(args.tmp_root, 'visualize', 'confusion', 'cad', 'prediction_seg.pdf'))

    return 1.0-subactivity_acc_ratio.avg


def parse_arguments():
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
        return x

    paths = config.Paths()

    # Path settings
    parser = argparse.ArgumentParser(description='CAD 120 dataset')
    parser.add_argument('--project-root', default=paths.project_root, help='intermediate result path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    parser.add_argument('--log-root', default=os.path.join(paths.log_root, 'cad120/parsing'), help='log files path')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/cad120/parsing'), help='path to latest checkpoint')
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize final results')

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=5e-4, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.8, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.8)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # i/o
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=1, help='Pre-fetching threads.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
