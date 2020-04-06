# -*- coding:utf-8 -*-
# @Time: 2020/2/12 15:37
# @Author: jockwang, jockmail@126.com
import logging
import os
import matplotlib.pyplot as plt

k = ['1', '2', '5', '10', '20', '50', '100']
modes = ['Precision', 'Recall', 'F1', 'HR', 'NDCG']

def addColor(labels, ax, dataset, data, mode='F1', K=k):
    # labels = ['Embedding', 'MLP', 'GAT', 'GCN', 'HNN', 'DEEP-HGCN', 'HGAT']
    # labels = ['DeepHGCN', 'HGCN', 'HNN', 'GAT', 'GCN', 'SHINE', 'DKN']
    marks = ['.', '+', 'v', 's', '*', 'o', 'x']
    ax.set_xticklabels(k)
    ax.set_xlabel('K', fontsize=20, fontweight='normal')
    ax.set_ylabel(mode+'@K', fontsize=20, fontweight='normal')
    ax.set_title(dataset, fontsize=20, fontweight='normal')
    for i in range(len(data)):
        ax.plot(K, data[i], label=labels[i], marker=marks[i])
    ax.legend(prop={'size': 10})

def draw():
    plt.style.use('ggplot')
    fig, axs = plt.subplots(len(modes), 1, figsize=(15, 30))
    plt.subplots_adjust(left=0.07, bottom=0.1, right=.93, top=.9, wspace=0.22)
    dataset = ['(a) Movie1M', '(b) Last.FM', '(c) Book Crossing']
    # 'DeepHGCN', 'HGCN', 'HNN', 'GAT', 'GCN', 'SHINE', 'DKN'
   
    # TODO: read the data from file (../log/TOP@K.txt)
    os.chdir('../')
    print(os.getcwd())
    topk_file = f'../log/TOP@K.txt'
    datasets, models, precision, recall, f1, hr, ndcg= [], [], [], [], [], [], []
    with open(topk_file, 'r', encoding='utf-8') as f:
        for line in f:
            array = line.strip().split(':')
            if array[0] not in datasets:
                datasets.append(array[0])
            if array[1] not in models:
                models.append(array[1])
            # if 'Precision' == array[-2]:
            #     eval(f'{array[1]}={dict()}')

    print(f'datasets({len(datasets)}):{datasets} models({len(models)}):{models}')
    f = open(topk_file, 'r', encoding='utf-8')
    labels = []
    precision_temp, recall_temp, f1_temp, hr_temp, ndcg_temp = [], [], [], [], []
    for dataset in datasets:
        for model in models:
            labels.append(model.upper())
            print(model)
            for line in f:
                array = line.strip().split(':')
                if model in line:
                    if array[2]:
                        if 'Precision' == array[-2]:
                            precision_temp.append(eval(array[-1]))
                        if 'Recall' == array[-2]:
                            recall_temp.append(eval(array[-1]))
                        if 'F-1' == array[-2]:
                            f1_temp.append(eval(array[-1]))
                        if 'HR' == array[-2]:
                            hr_temp.append(eval(array[-1]))
                        if 'NDCG' == array[-2]:
                            ndcg_temp.append(eval(array[-1]))

            # print(precision_temp[-1])
            precision.append(precision_temp[-1])
            recall.append(recall_temp[-1])
            f1.append(f1_temp[-1])
            hr.append(hr_temp[-1])
            ndcg.append(ndcg_temp[-1])

        data_eval = [precision, recall, f1, hr, ndcg]
        for k, i in enumerate(modes):
            addColor(labels, axs[k], dataset=dataset.title(), mode=i, data=data_eval[k])
        plt.savefig(f'../log/figure/{dataset.title()}_figure.png',format='png')
        print(f'save figure to ../log/figure/{dataset.title()}_eval_figure.top@k.png')
        # plt.show()
    f.close()
##############################
# drawing top@k of each epoch
##############################
def add_color_each_mode(ax, dataset, data, auc, K=k):
    labels = ['Precision@K', 'Recall@K', 'F-1@K', 'HR@K', 'nDCG@K']
    marks = ['o', '+', 'v', 's', '*']
    ax.set_xticklabels(k)
    ax.set_xlabel(f'K\n\nAUC:{auc:.5f}', fontsize=20, fontweight='normal')
    ax.set_ylabel('TOP@K', fontsize=20, fontweight='normal')
    ax.set_title(dataset, fontsize=20, fontweight='normal')
    for i in range(len(data)):
        ax.plot(K, data[i], label=labels[i], marker=marks[i])
    ax.legend(prop={'size': 15})

def draw_each_mode(args, data_list, auc, save_path, is_maxauc, is_show=False):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    plt.subplots_adjust(left=0.07, bottom=0.1, right=.93, top=.9, wspace=0.22)
    add_color_each_mode(axs, dataset=args.dataset, data=data_list, auc=auc)
    if is_maxauc:
        plt.savefig(save_path + f'{args.dataset}_{args.model}_lr{args.lr}_weight_decay{args.l2_weight_decay}_eval_figure.best_auc.png',format='png')
        logging.info(f'save figure to save_path {args.dataset}_{args.model}_lr{args.lr}_weight_decay{args.l2_weight_decay}_eval_figure.best_auc.png')
    else:
        plt.savefig(save_path + f'{args.dataset}_{args.model}_lr{args.lr}_weight_decay{args.l2_weight_decay}_eval_figure.eraly_stop.png',format='png')
        logging.info(f'save figure to {save_path}/{args.dataset}_{args.model}_lr{args.lr}_weight_decay{args.l2_weight_decay}_eval_figure.eraly_stop.png')
    if is_show:
        plt.show()
if __name__ == '__main__':
    # data_list = [[0.04, 0.04, 0.024000000000000004, 0.02, 0.014499999999999999, 0.008600000000000002, 0.005699999999999999]
    #             ,[0.012833333333333332, 0.022666666666666665, 0.041833333333333333, 0.06726190476190476, 0.10209523809523809, 0.16159523809523813, 0.21895238095238093]
    #             ,[0.01943217665615142, 0.02893617021276596, 0.03050126582278481, 0.030832196452933152, 0.025393506228303043, 0.016330880501384967, 0.011110753121224323]
    #             ,[0.014388489208633094, 0.02877697841726619, 0.04316546762589928, 0.07194244604316546, 0.10431654676258993, 0.15467625899280577, 0.20503597122302158]
    #             ,[0.04, 0.06523719014285831, 0.0778778713685842, 0.09440938409445768, 0.1062567188297546, 0.12814949248538698, 0.14152511177580518]]
    # draw_each_mode(dataset='music', data_list=data_list, is_maxauc=True, save_path='../log/figure/', is_show=True)
    draw()