"""
Created on Jan 28, 2018

@author: Siyuan Qi

Description of the file.

"""

import itertools

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, filename=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    ax = plt.gca()
    ax.tick_params(axis=u'both', which=u'both', length=0)
    # matplotlib.rcParams.update({'font.size': 15})
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]), verticalalignment='center', horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def plot_segmentation(input_labels_list, endframe, vmax=None, filename=None, border=True, cmap=plt.get_cmap('gist_rainbow')):
    plt_idx = 0
    aspect_ratio = 30
    fig = plt.figure(figsize=(28, 5))
    for input_labels in input_labels_list:
        seg_image = np.empty((int(endframe/aspect_ratio), endframe))

        for frame in range(endframe):
            seg_image[:, frame] = input_labels[frame]

        plt_idx += 1
        ax = plt.subplot(len(input_labels_list), 1, plt_idx)
        if not border:
            ax.axis('off')
        if vmax:
            ax.imshow(seg_image, vmin=0, vmax=vmax, cmap=cmap)
        else:
            ax.imshow(seg_image, cmap=cmap)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def main():
    pass


if __name__ == '__main__':
    main()
