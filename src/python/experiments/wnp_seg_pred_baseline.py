"""
Created on Feb 03, 2018

@author: Siyuan Qi

Description of the file.

"""

# System imports
import os
import time
import datetime
import argparse
import decimal

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
import parser.grammarutils

cross_entropy = torch.nn.CrossEntropyLoss().cuda()
mse_loss = torch.nn.MSELoss().cuda()
softmax = torch.nn.Softmax(dim=2)
ctc_loss = warpctc_pytorch.CTCLoss().cuda()


def loss_func(model_outputs, labels, probs, total_lengths, args):
    loss = 0
    for i_batch in range(model_outputs.size()[1]):
        gt_pred_labels = list()
        seg_length = int(total_lengths[i_batch])
        current_label = int(labels[0, i_batch])
        for f in range(seg_length):
            if int(labels[f, i_batch]) != current_label:
                current_label = int(labels[f, i_batch])
                gt_pred_labels.extend([current_label for _ in range(f-len(gt_pred_labels)-1)])
        gt_pred_labels.extend([int(labels[seg_length-1, i_batch]) for _ in range(seg_length-len(gt_pred_labels))])
        gt_pred_labels = utils.to_variable(torch.LongTensor(gt_pred_labels), args.cuda)

        loss += cross_entropy(model_outputs[:seg_length, i_batch, :], gt_pred_labels)
        # loss += mse_loss(model_outputs[:seg_length, i_batch, :], probs[:seg_length, i_batch, :])
    return loss


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))

    # Load data
    training_set, testing_set, train_loader, test_loader = utils.get_wnp_data(args)
    features, labels, seg_lengths, total_length, activity, sequence_id = training_set[0]
    feature_size = features.shape[1]
    label_num = len(datasets.wnp_metadata.subactivities)
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
        epoch_error = validate(test_loader, model, args=args)

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
    subactivity_acc_ratio = logutil.AverageMeter()

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

        model_outputs = model(features)
        _, pred_labels = torch.max(model_outputs, dim=2)
        train_loss = criterion(model_outputs, labels, probs, total_lengths, args)

        # Log
        losses.update(train_loss.data[0], torch.sum(total_lengths).data[0])

        subact_micro_result = sklearn.metrics.precision_recall_fscore_support(labels.cpu().data.numpy().flatten().tolist(), pred_labels.cpu().data.numpy().flatten().tolist(), labels=range(10), average='micro')
        subactivity_acc_ratio.update(subact_micro_result[0], torch.sum(total_lengths).data[0])

        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        break

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)

    print('Epoch: [{0}] Avg Subactivity Accuracy Ratio {act_err.avg:.3f}; Average Loss {losses.avg:.3f}; Batch Avg Time {b_time.avg:.3f}'
          .format(epoch, act_err=subactivity_acc_ratio, losses=losses, b_time=batch_time))


def get_gt_pred(model_outputs, labels, total_lengths):
    all_gt_pred_labels = list()
    for i_batch in range(model_outputs.size()[1]):
        gt_pred_labels = list()
        seg_length = int(total_lengths[i_batch])
        current_label = int(labels[0, i_batch])
        for f in range(seg_length):
            if int(labels[f, i_batch]) != current_label:
                current_label = int(labels[f, i_batch])
                gt_pred_labels.extend([current_label for _ in range(f-len(gt_pred_labels)-1)])
        gt_pred_labels.extend([int(labels[seg_length-1, i_batch]) for _ in range(seg_length-len(gt_pred_labels))])
        all_gt_pred_labels.extend(gt_pred_labels)
    return all_gt_pred_labels


def validate(val_loader, model, args, test=False):
    def compute_accuracy(gt_results, results, metric='micro'):
        return sklearn.metrics.precision_recall_fscore_support(gt_results, results, labels=range(10), average=metric)

    batch_time = logutil.AverageMeter()
    subactivity_acc_ratio = logutil.AverageMeter()

    all_gt_seg_predictions = list()
    all_seg_predictions = list()

    # switch to evaluate mode
    model.eval()

    end_time = time.time()
    for i, (features, labels, probs, total_lengths, ctc_labels, ctc_lengths, activities, sequence_ids) in enumerate(val_loader):
        features = utils.to_variable(features, args.cuda)
        labels = utils.to_variable(labels, args.cuda)

        total_lengths = torch.autograd.Variable(total_lengths)

        # Inference
        model_outputs = model(features)
        gt_pred_labels = get_gt_pred(model_outputs, labels, total_lengths)
        _, pred_labels = torch.max(model_outputs, dim=2)
        pred_labels = pred_labels.cpu().data.numpy().flatten().tolist()
        all_gt_seg_predictions.extend(gt_pred_labels)
        all_seg_predictions.extend(pred_labels)

        # Evaluation
        # Segment prediction
        subact_micro_result = compute_accuracy(gt_pred_labels, pred_labels)
        subactivity_acc_ratio.update(subact_micro_result[0], torch.sum(total_lengths).data[0])

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    print(' * Avg Subactivity Accuracy Ratio {act_err.avg:.3f}; Batch Avg Time {b_time.avg:.3f}'
          .format(act_err=subactivity_acc_ratio, b_time=batch_time))
    print(compute_accuracy(all_gt_seg_predictions, all_seg_predictions, metric='macro'))
    return 1.0 - subactivity_acc_ratio.avg


def parse_arguments():
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
        return x

    paths = config.Paths()

    # Path settings
    default_setting = 'all'
    parser = argparse.ArgumentParser(description='Watch-n-patch dataset')
    parser.add_argument('--setting', default=default_setting, help='kitchen or office')
    parser.add_argument('--project-root', default=paths.project_root, help='intermediate result path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    parser.add_argument('--log-root', default=os.path.join(paths.log_root, 'wnp/baseline/segment'), help='log files path')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/wnp/baseline/segment'), help='path to latest checkpoint')
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize final results')

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=8e-4, metavar='LR',
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