import argparse
import logging
import time

from train import *
from utils import seed_everything




if __name__ == '__main__':
    seed_everything(seed=1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='music',
                        help='which dataset to train(dataset: book, movie1m, music, restaurant, movie20m)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='set the batch size of training')
    parser.add_argument('-m', '--model', type=str, default='gat',help='which module to choice(module: gcn, hgcn...)')
    parser.add_argument('-D', '--dim', type=int, default=16, help='embedding dimension')
    parser.add_argument('-h1', '--hidden1', type=int, default=64, help='the hidden_1 is u_hidden_size')
    parser.add_argument('-h2', '--hidden2', type=int, default=32, help='the hidden_2 is i_hidden_size')
    parser.add_argument('-H', '--heads', type=int, default=6, help="the number of GAT'heads")
    parser.add_argument('-C', '--cat', type=int, default=1, help="Is Inner product(0) or Concatenation(1)？")

    parser.add_argument('-ci', '--c_in', type=float, default=1., help='the curvature in for hnn(c) and hgcn')
    parser.add_argument('-co', '--c_out', type=float, default=1.0, help='the curvature out for hgcn')
    parser.add_argument('-do', '--dropout', type=float, default=0., help='the dropout for hyperbolic param')
    parser.add_argument('-E', '--epochs', type=int, default=300, help='the epochs of the training')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='learning rate') # 1e-4
    parser.add_argument('-w', '--l2_weight_decay', type=float, default=2e-4, help='l2_weight_decay')
    parser.add_argument('-e', '--early_stop_patience', type=int, default=10, help='the parameter for stopping early')
    parser.add_argument('-c', '--cuda', type=int, default=1, help='1:use cuda;0:non-use cuda')
    parser.add_argument('-ctr', '--show_ctr', type=int, default=1, help="1:show ctr;0:don't show ctr")
    parser.add_argument('-topk', '--show_topk', type=int, default=0, help="1:show topk;0:don't show topk")
    parser.add_argument('-time', '--show_time', type=int, default=1, help="1:show time;0:don't show time")
    parser.add_argument('-M', '--mode', type=str, default='train', help="train model;load model")
    args = parser.parse_args()
    t = time.time()
    # try:
    main(args)
    # except Exception as e:
    #     logging.error(e)
    logging.info('Time used: %ds\n\n\n==========================================================================================\n' % (time.time() - t))
