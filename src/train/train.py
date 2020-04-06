import time
import logging

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from eval.ctr_eval import ctr_eval
from eval.eval_utils import topk_settings
from train.choice_model import choice_model
# from train.Model import Model
from train.train_utils import *

from hyper_layers.hyper_utils import printModelParameters

from utils import *

########################
# mode: train and evaluate
########################
def train(args, number):
    # other parameter .........
    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cur_gpu_info = gpu_empty_info()
        logging.info('Current %s' % cur_gpu_info[0])
    else:
        device = torch.device('cpu')
    logging.info('Use device:'+str(device))
    # Early stop and checkpoint
    early_stopping = EarlyStoppingCriterion(patience=args.early_stop_patience) ## initialize the early_stopping object
    stop_epoch, auc_max = 0, 0.0
    # graph
    graph = get_graph(args.dataset, number).to(device)
    logging.info(graph)
    # data loader
    train_loader, test_loader, valid_loader = data_loader(args.dataset, args.batch_size, number)
    # param
    u_nodes, i_nodes = number[args.dataset]['users'], number[args.dataset]['entities']
    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(train_loader, test_loader)

    # choice the model and optimizer
    model, optimizer = choice_model(args, u_nodes, i_nodes, device)
    # model = Model(number, args)
    logging.info(model)
    # optimizer = model.optimizer
    logging.info(optimizer)
    # criterion = model.loss
    criterion = nn.BCELoss()

    # epoch
    start_total_time = time.time()
    each_epoch_time = []
    for epoch in tqdm(range(args.epochs)):
        start_epoch_time = time.time()
        running_train_loss = 0
        train_losses = []
        valid_losses = []
        test_losses = []
        maximum_auc_topk = []
        for k, [user, item_hot, label, _] in enumerate(train_loader):
            u, i, l = user.to(device), item_hot.to(device), label.to(device)

            optimizer.zero_grad()
            out = model(u, i, graph).double()
            loss = criterion(out, l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_train_loss += loss.item()
            train_losses.append(loss.item())

            if k % 50 == 49:
                # compute the gpu usage and report in the logging
                if args.cuda:
                    try:
                        usage_str = gpu_usage(cur_gpu_info[1])
                    except Exception as e:
                        usage_str = f'GPU Error: [{e}] Used CPU'
                else:
                    usage_str = 'CPU'

                # valid loss
                for user, item_hot, label, _ in valid_loader:
                    u, i, l = user.to(device), item_hot.to(device), label.to(device)
                    out = model(u, i, graph).double()
                    loss = criterion(out, l)
                    valid_losses.append(loss.item())

                # test loss
                for user, item_hot, label, _ in test_loader:
                    u, i, l = user.to(device), item_hot.to(device), label.to(device)
                    out = model(u, i, graph).double()
                    loss = criterion(out, l)
                    test_losses.append(loss.item())

                logging.info(f'[{args.dataset}:{args.model}{args.dim}]Epoch:{epoch + 1} Step:{k + 1} '
                             f'train loss:{running_train_loss / 50:.5f} valid loss:{np.average(valid_losses):.5f} '
                             f'test loss:{np.average(test_losses):.5f}| {usage_str}')
                running_train_loss = 0

        ## early stop and save model
        # early_stop_loss = (np.mean(train_losses)+np.average(valid_losses))/2
        early_stop_loss = np.average(test_losses)
        # early_stop_loss = np.average(valid_losses)
        early_stopping(early_stop_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping. Epochs:%d early_stop_loss:%.6f" % (epoch + 1, early_stop_loss))
            stop_epoch = epoch + 1
            save_model(args, model, 0)
            break
        end_epoch_time = time.time()
        # each epoch
        each_epoch_time.append(float(end_epoch_time - start_epoch_time))
        logging.info(f'[{args.dataset}:{args.model}{args.dim}]Epoch:{epoch + 1} Cost time:{each_epoch_time[-1]:.3f}s')
        # evaluation
        start_eva_time = time.time()
        logging.info('Evaluation...')
        if args.show_ctr:
            auc, recall, f1 = ctr_eval(test_loader, device, model, graph)
            logging.info(f'AUC={auc:.4f} Recall={recall:.4f} F-1={f1:.4f}')
            # save model, checkpoint: maximum auc
            if auc > auc_max and auc > 0.80:
                auc_max = auc
                logging.info(f'Maximum AUC: {auc} Saving model...')
                t_k = topk(args, device, model, graph, user_list, train_record,test_record, item_set, i_nodes ,k_list, auc_max,
                     is_maxauc=True)
                maximum_auc_topk.append((auc, t_k))
                save_topk(args, maximum_auc_topk[-1][0], maximum_auc_topk[-1][1], early_stopping.early_stop)
                save_model(args, model, auc_max)
        if args.show_topk:
            topk(args, device, model, graph, user_list, train_record, test_record, item_set, i_nodes, k_list, auc)
        end_eva_time = time.time()
        # compute and report the total time used for test set evaluation
        logging.info(f'[{args.dataset}:{args.model}{args.dim}]Epoch:{epoch + 1} Evaluation cost total time:{float(end_eva_time - start_eva_time):.3f}s')
        # early_stopping needs the loss to check if it has decresed, valid data:to stop
        logging.info('Early stopping using valid data...')
    # stop
    if not args.show_topk:
        topk(args, device, model, graph, user_list, train_record, test_record, item_set, i_nodes, k_list, auc)

    # compute and report the total time used for one epoch and average time used for each epoch
    end_total_time = time.time()
    for epoch, t in enumerate(each_epoch_time):
        logging.info(f'[{args.dataset}:{args.model}{args.dim}]Epoch:%d Cost time:%.3fs' % (epoch + 1, t))
    total_time = float(end_total_time - start_total_time)
    logging.info(f'[{args.dataset}:{args.model}{args.dim}]Training cost total time:%.3fs Average time:%.3fs' % (total_time, total_time / stop_epoch))

########################
# mode: load and evaluate
########################
def load_and_eval(args, number):
    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cur_gpu_info = gpu_empty_info()
        logging.info('Current %s' % cur_gpu_info[0])
    else:
        device = torch.device('cpu')
    logging.info('Use device:'+str(device))
    # graph
    graph = get_graph(args.dataset, number).to(device)
    logging.info(graph)
    # data loader
    train_loader, test_loader, valid_loader = data_loader(args.dataset, args.batch_size, number)
    # params
    u_nodes, i_nodes = number[args.dataset]['users'], number[args.dataset]['entities']
    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(train_loader, test_loader, number[args.dataset]['items'])
    # load model
    model = choice_model(args,u_nodes, i_nodes, device)
    model_params_file = f'../model/{args.dataset}_{args.model}_dim{args.dim}_lr.{args.lr}_weight_decay.{args.l2_weight_decay}_params_earlystop.pkl'
    # model_params_file = '../model/checkpoint.pt'
    model = read_model(model, model_params_file)
    # evaluation
    auc, recall, f1 = ctr_eval(test_loader, device, model, graph)
    logging.info(f'AUC={auc:.4f} Recall={recall:.4f} F-1={f1:.4f}')
    topk(args, device, model, graph, user_list, train_record, test_record, item_set, i_nodes, k_list, auc)

    printModelParameters(model)
########################
# main function
########################
def main(args):
    logging.info(f'save debug info to {logging_setting(args)}')
    logging.info(args)
    torch.autograd.set_detect_anomaly(True)
    
    number = get_number(args)
    if not number:
        return
    if args.mode == 'train':
        train(args, number)
    elif args.mode == 'load':
        load_and_eval(args, number)
    else:
        logging.ERROR('No such mode. There are two modes:--mode ["train" or "load"]')
        return


