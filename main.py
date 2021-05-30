import numpy as np
from matplotlib import pyplot as plt

import train
import analysis


def plot_losses(losses, dataset='train1'):
    plt.semilogy(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss in log')
    plt.title(f'Semilogy Loss - {dataset}')
    plt.show()


def train_1_eval_test_create_comp1(print_acc=True):
    test_path = f'/datashare/hw1/test1.wtag'
    reg, threshold = 1., 1.0  # 0.96321
    weights, features, losses = train.train('train1', 1000, threshold, reg)
    weights = np.array(weights)
    analysis.create_comp_file('comp1.wtag', f'/datashare/hw1/comp1.words', weights, features)
    if print_acc:
        acc = analysis.find_acc(test_path, weights, features)
        print(f'Accuracy of train 1 test is {acc: .5f}')
    # plot_losses(losses, 'train1')


def train_2_create_comp2():
    reg, threshold = 1., 1.0  # 0.96321
    weights, features, losses = train.train('train2', 1000, threshold, reg)
    analysis.create_comp_file('comp2.wtag', f'/datashare/hw1/comp2.words', weights, features)
    # plot_losses(losses, 'train2')


def main():
    train_1_eval_test_create_comp1(print_acc=True)
    train_2_create_comp2()


if __name__ == '__main__':
    main()

