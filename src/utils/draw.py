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
    # music
    data_eval = [
        # pre
         # Embedding
        [[0.0, 0.005, 0.01, 0.013000000000000003, 0.0075, 0.0048, 0.0038],
         # MLP
         [0.0, 0.015, 0.012000000000000002, 0.011000000000000001, 0.008, 0.0046, 0.0039000000000000003],
         # GAT
         [0.0, 0.015, 0.01, 0.012000000000000002, 0.008, 0.0046, 0.0041],
         # GCN
         [0.03, 0.015, 0.014000000000000002, 0.01, 0.006500000000000001, 0.0038, 0.0030000000000000005],
         # HNN
         [0.01, 0.015, 0.012000000000000002, 0.008, 0.007000000000000001, 0.0048, 0.0036],
         # DEEP-HGCN
         [0.01, 0.005, 0.002, 0.002, 0.002, 0.0036, 0.003],
         # HGAT
         [0.01, 0.005, 0.002, 0.005, 0.0035000000000000005, 0.0044, 0.0036000000000000003]
         ],
        # recall
        [[0.0, 0.003333333333333333, 0.023666666666666662, 0.04759523809523809, 0.055095238095238086, 0.09202380952380952, 0.14935714285714285],
         [0.0, 0.008666666666666666, 0.025095238095238094, 0.04176190476190476, 0.05709523809523809, 0.08502380952380953, 0.14302380952380955],
         [0.0, 0.008666666666666666, 0.015095238095238092, 0.04592857142857142, 0.05709523809523809, 0.08502380952380953, 0.15502380952380954],
         [0.008666666666666666, 0.008666666666666666, 0.025928571428571426, 0.032928571428571425, 0.04126190476190476, 0.05892857142857142, 0.13402380952380955],
         [0.003333333333333333, 0.008666666666666666, 0.024499999999999997, 0.032, 0.04659523809523809, 0.09192857142857143, 0.13685714285714284],
         [0.0025, 0.0025, 0.0025, 0.0125, 0.018333333333333333, 0.06109523809523809, 0.09519047619047619],
         [0.0025, 0.0025, 0.0025, 0.011166666666666667, 0.024499999999999997, 0.10033333333333333, 0.15642857142857144]
         ],
        # f1
        [[0.0, 0.004, 0.014059405940594058, 0.020422003929273087, 0.01320273868391023, 0.009124084001377071, 0.007411435500419738],
         [0.0, 0.010985915492957746, 0.01623620025673941, 0.017413357400722022, 0.014033650329188001, 0.00872780404866904, 0.007592953911972518],
         [0.0, 0.010985915492957746, 0.012030360531309297, 0.019028360049321826, 0.014033650329188001, 0.00872780404866904, 0.007988717979411062],
         [0.013448275862068964, 0.010985915492957746, 0.018182468694096603, 0.015341098169717137, 0.011230807577268197, 0.00713960373491232, 0.005868635968722851],
         [0.005000000000000001, 0.010985915492957746, 0.016109589041095895, 0.012799999999999999, 0.012171479342514441, 0.009123615418697386, 0.007015459723352319],
         [0.004, 0.0033333333333333335, 0.0022222222222222222, 0.0034482758620689655, 0.00360655737704918, 0.006799352274400117, 0.005816682832201745],
         [0.004, 0.0033333333333333335, 0.0022222222222222222, 0.006907216494845361, 0.006125, 0.008430299172501592, 0.007038028923406535]
         ],
        # hr
        [[0.0, 0.0035971223021582736, 0.017985611510791366, 0.046762589928057555, 0.0539568345323741, 0.08633093525179857, 0.1366906474820144],
        [0.0, 0.01079136690647482, 0.02158273381294964, 0.039568345323741004, 0.05755395683453238, 0.08273381294964029, 0.14028776978417265],
        [0.0, 0.01079136690647482, 0.017985611510791366, 0.04316546762589928, 0.05755395683453238, 0.08273381294964029, 0.1474820143884892],
        [0.01079136690647482, 0.01079136690647482, 0.025179856115107913, 0.03597122302158273, 0.046762589928057555, 0.0683453237410072, 0.1079136690647482],
        [0.0035971223021582736, 0.01079136690647482, 0.02158273381294964, 0.02877697841726619, 0.050359712230215826, 0.08633093525179857, 0.12949640287769784],
        [0.0035971223021582736, 0.0035971223021582736, 0.0035971223021582736, 0.007194244604316547, 0.014388489208633094, 0.06474820143884892, 0.1079136690647482],
        [0.0035971223021582736, 0.0035971223021582736, 0.0035971223021582736, 0.017985611510791366, 0.025179856115107913, 0.07913669064748201, 0.12949640287769784]
         ],
        # ndcg
        [[0.0, 0.006309297535714575, 0.024922828697182434, 0.04631880074959125, 0.05137527142753082, 0.06509432315113649,  0.07925588945011085],
        [0.0, 0.018927892607143726, 0.03053347682417997, 0.042734817752838744, 0.055142492142097634, 0.0657018914959692, 0.08119290808842881],
        [0.0, 0.018927892607143726, 0.027103186260223077, 0.04461199402000124, 0.0547249353758804, 0.06563567398258843, 0.08354472594059706],
        [0.03, 0.03, 0.04167874270842689, 0.05079337256811798, 0.05864464107784664, 0.06553296673934363, 0.08104611601998181],
        [0.01, 0.02261859507142915, 0.02999057219912211, 0.03711471594128255, 0.051976226998019556, 0.06784969914869322, 0.08106594343965201],
        [0.01, 0.01, 0.01, 0.013010299956639812, 0.01829972941315111, 0.044924343434104064, 0.05788226235015648],
        [0.01, 0.01, 0.01, 0.021826593557393924, 0.02731840455817842, 0.05553665403267187, 0.07394500169912774]
         ]
    ]
    # TODO: read the data from file (../log/TOP@K.txt)
    os.chdir('../')
    print(os.getcwd())
    topk_file = f'../log/TOP@K.txt'
    datasets, models = [], []
    precision_best_auc, recall_best_auc, f1_best_auc, hr_best_auc, ndcg_best_auc = [], [], [], [], []
    precision_temp, recall_temp, f1_temp, hr_temp, ndcg_temp = [], [], [], [], []
    precision_early_stop, recall_early_stop, f1_early_stop, hr_early_stop, ndcg_early_stop = [],[],[],[],[]
    with open(topk_file, 'r', encoding='utf-8') as f:
        for line in f:
            array = line.strip().split(':')
            if array[0] not in datasets:
                datasets.append(array[0])
            if array[1] not in models:
                models.append(array[1])
            # not is maximum auc
            if eval(array[2]):
                precision_temp.append(eval(array[4]))
                recall_temp.append(eval(array[5]))
                f1_temp.append(eval(array[6]))
                hr_temp.append(eval(array[7]))
                ndcg_temp.append(eval(array[8]))
            else:
                precision_best_auc.append(precision_temp[-1])
                recall_best_auc.append(recall_temp[-1])
                f1_best_auc.append(f1_temp[-1])
                hr_best_auc.append(hr_temp[-1])
                ndcg_best_auc.append(ndcg_temp[-1])

                precision_early_stop.append(eval(array[4]))
                recall_early_stop.append(eval(array[5]))
                f1_early_stop.append(eval(array[6]))
                hr_early_stop.append(eval(array[7]))
                ndcg_early_stop.append(eval(array[8]))


    print(f'datasets({len(datasets)}):{datasets} models({len(models)}):{models}')
    data_eval_early_stop = [precision_early_stop, recall_early_stop, f1_early_stop, hr_early_stop, ndcg_early_stop]
    data_eval_best_auc = [precision_best_auc, recall_best_auc, f1_best_auc, hr_best_auc, ndcg_best_auc]
    # labels setting
    labels = [i.upper() for i in models]
    # Early stop
    for dataset in datasets:
        for k, i in enumerate(modes):
            addColor(labels, axs[k], dataset=dataset.title(), mode=i, data=data_eval_early_stop[k])
        plt.savefig(f'../log/figure/{dataset.title()}_eval_figure_early_stop.top@k.png',format='png')
        print(f'save figure to ../log/figure/{dataset.title()}_eval_figure_early_stop.top@k.png')
        plt.show()

    # Best AUC
    for dataset in datasets:
        for k, i in enumerate(modes):
            addColor(labels, axs[k], dataset=dataset.title(), mode=i, data=data_eval_best_auc[k])
        plt.savefig(f'../log/figure/{dataset.title()}_eval_figure_best_auc.top@k.png',format='png')
        print(f'save figure to ../log/figure/{dataset.title()}_eval_figure_best_auc.top@k.png')
        # plt.show()


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
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if is_maxauc:
        plt.savefig(save_path + f'{args.dataset}_{args.model}_lr{args.lr}_weight_decay{args.l2_weight_decay}_eval_figure.best_auc.png',format='png')
        logging.info(f'save figure to save_path {args.dataset}_{args.mode}_eval_figure.best_auc.png')
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