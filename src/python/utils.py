"""
Created on Jan 11, 2018

@author: Siyuan Qi

Description of the file.

"""

import os

import numpy as np
import torch
import torch.utils.data

import datasets


def to_variable(v, use_cuda):
    if use_cuda:
        v = v.cuda()
    return torch.autograd.Variable(v)


def get_cad_data(args):
    training_set = datasets.CAD120(os.path.join(args.tmp_root, 'data', 'cad_training.p'))
    testing_set = datasets.CAD120(os.path.join(args.tmp_root, 'data', 'cad_testing.p'))

    train_loader = torch.utils.data.DataLoader(training_set, collate_fn=datasets.utils.collate_fn_cad,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testing_set, collate_fn=datasets.utils.collate_fn_cad,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)

    print('Dataset sizes: {} training, {} testing.'.format(len(train_loader), len(test_loader)))
    return training_set, testing_set, train_loader, test_loader


def get_wnp_data(args):
    training_set = datasets.WNP(os.path.join(args.tmp_root, 'data', 'wnp_{}_training.pkl'.format(args.setting)))
    testing_set = datasets.WNP(os.path.join(args.tmp_root, 'data', 'wnp_{}_testing.pkl'.format(args.setting)))

    train_loader = torch.utils.data.DataLoader(training_set, collate_fn=datasets.utils.collate_fn_wnp,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testing_set, collate_fn=datasets.utils.collate_fn_wnp,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)

    print('Dataset sizes: {} training, {} testing.'.format(len(train_loader), len(test_loader)))
    return training_set, testing_set, train_loader, test_loader


def main():
    pass


if __name__ == '__main__':
    main()
