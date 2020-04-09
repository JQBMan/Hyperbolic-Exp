import torch
import logging
import pandas as pd
from torch_geometric.data import Data
import pynvml
import numpy as np
import random
import os

#####################
# get_graph function
#####################
def get_graph(dataset, number):
    df = pd.read_csv('../data/'
                     + dataset + '/kg_final.txt', sep='\t', header=None, index_col=None)
    logging.info('Generating subgraph...')
    head_index, tail_index = [], []

    for value in df.values:
        head_index.append(value[0])
        tail_index.append(value[-1])
    edge_index = torch.tensor([head_index, tail_index], dtype=torch.long)
    x = torch.tensor(range(number[dataset]['entities']), dtype=torch.long)
    logging.info('Done.')
    return Data(x=x, edge_index=edge_index)
########################
# set the random seed
########################
def seed_everything(seed=5555):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################
# gpu info function
########################
def gpu_empty_info():
    try:
        # empty the gpu cache
        torch.cuda.empty_cache()
        current_gpu_info = gpu_info()
        info_str = ''
        for key in current_gpu_info:
            info_str += "GPU_{}: Used:".format(key) + '{:.3f}'.format(current_gpu_info[key]['used']) +\
                "GB/" + '{:.3f}'.format(current_gpu_info[key]['total']) +\
                "GB Percent:" + '{:.2f}'.format(current_gpu_info[key]['percent']) + '% '
        return [info_str, current_gpu_info]
    except:
        return [info_str, ]
def gpu_usage(gpu_info_dit):
    usage_str = ''
    try:
        gpu_usage_info = gpu_info()  # info list
        for key in gpu_usage_info:
            usage_str += "GPU_{}: Used:".format(key) + '{:.3f}'.format(
                gpu_usage_info[key]['used'] - gpu_info_dit[key]['used']) + \
                         "GB/" + '{:.3f}'.format(gpu_usage_info[key]['total']) + \
                         "GB Percent:" + '{:.2f}'.format(
                gpu_usage_info[key]['percent'] - gpu_info_dit[key]['percent']) + '% '
        return usage_str
    except:
        return usage_str
def gpu_info():
    try:
        pynvml.nvmlInit()
        # print("Driver Version:", pynvml.nvmlSystemGetDriverVersion()) #gpu version
        deviceCount = pynvml.nvmlDeviceGetCount() #gpu count
        gpu_info = {}
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("Device", i, ":", pynvml.nvmlDeviceGetName(handle)) #gpu name
            gpu_info[str(i)] = {
                    'version':str(pynvml.nvmlSystemGetDriverVersion()),
                    'name':str(pynvml.nvmlDeviceGetName(handle)),
                    'used':meminfo.used / 1024 /1024 / 1024,
                    'free':meminfo.free / 1024 /1024 / 1024,
                    'total':meminfo.total / 1024 /1024 / 1024,
                    'percent':(meminfo.used)/(meminfo.total)*100
                }
            pynvml.nvmlShutdown()
        return gpu_info
    except Exception as e:
        logging.info(f'GPU Erorr:[{e}]')




#####################
# get_count function
#####################
# get the count of the follow parameters:
# user_count, item_count, entity_count and relations in ratings_final.txt and kg_final.txt
def get_count(DATASET):
    # parameters
    users = set()
    items = set()
    # item_cnt
    interaction_cnt = 0
    kg_triple_cnt = 0
    relations = set()

    # file path
    ratings_final_file = '../data/' + DATASET + '/ratings_final.txt'
    kg_final_file = '../data/' + DATASET + '/kg_final.txt'

    # deal...
    logging.info('processing the ratings_final file(is %s)...' % ratings_final_file)
    try:
        with open(ratings_final_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                interaction_cnt += 1
                line = line.strip('\n').split('\t')
                users.add(line[0])
                items.add(line[1])

        # output: count(users) & count(items)
        logging.info('users: %d' % len(users))
        logging.info('items: %d' % len(items))
        logging.info('interactions: %d' % interaction_cnt)
        item_cnt = len(items)
        entity_cnt = item_cnt

        logging.info('processing the kg_final file(is %s)...' % kg_final_file)
        with open(kg_final_file, 'r', encoding='utf-8') as kf:
            for line in kf:
                kg_triple_cnt += 1
                line = line.strip('\n').split('\t')
                head = line[0]
                relations.add(line[1])
                tail = line[2]
                if head not in items:
                    entity_cnt += 1
                    items.add(head)
                if tail not in items:
                    entity_cnt += 1
                    items.add(tail)

        # output: count(entities) & count(relations)
        logging.info('entities: %d' % entity_cnt)
        logging.info('relations: %d' % len(relations))
        logging.info('KG triples: %d\n' % kg_triple_cnt)

        return {
            'users': len(users),
            'items': item_cnt,
            'interaction': interaction_cnt,
            'entities': entity_cnt,
            'relations': len(relations),
            'kg_triples': kg_triple_cnt,
            'hyperbolicity': 0 # computing using hypuni.get_hyperbolicity()
        }
    except:
        logging.ERROR('No such file or directory: %s' % ratings_final_file)
#
def get_number(args):
    number = {}
    try:
        number[args.dataset] = get_count(args.dataset)
    except Exception as e:
        number = None
        logging.ERROR(f'No such dataset: {args.dataset}: Error[{e}]')
    finally:
        logging.info('NUMBER:' + str(number))
        return number