import argparse
import numpy as np
from numpy import random

from utils import get_count

DATASETS = ['book', 'movie1m', 'music', 'restaurant', 'movie20m']
# user_id   entity_id   ratings
RATING_FILE_NAME = dict({
    'amazon-book': 'BX-Book-Ratings.csv',
    'last-fm': 'user_artists.dat',
    'movie1m': 'ratings.dat',
    'movie20m':'ratings.csv',
    'book': 'BX-Book-Ratings.csv',
    'news': 'ratings.txt',
    'music': 'user_artists.dat',
    'restaurant': ''})
SEP = dict({'amazon-book': ';', 'movie1m': '::', 'movie20m': ',', 'book': ';', 'news': '\t', 'music': '\t', 'last-fm': '\t'})
THRESHOLD = dict({'amazon-book': 0, 'movie1m': 4, 'movie20m': 4, 'book': 0, 'news': 0, 'music': 0, 'last-fm': 0})

def read_item_index_to_entity_id_file():
    file = '../data/' + dataset + '/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        try:
            item_index = line.strip().split('\t')[0]
            satori_id = line.strip().split('\t')[1]
            item_index_old2new[item_index] = i
            entity_id2index[satori_id] = i
            i += 1
        except:
            print('the line passed: %s' % line)
            continue

def convert_rating():
    file = '../data/' + dataset + '/' + RATING_FILE_NAME[dataset]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        # if 'userId' in line:
        #     continue
        try:
            array = line.strip().split(SEP[dataset])

            # remove prefix and suffix quotation marks for BX dataset
            if dataset in ['book', 'amazon-book']:
                array = list(map(lambda x: x[1:-1], array))

            item_index_old = array[1]
            if item_index_old not in item_index_old2new:  # the item is not in the final item set
                continue
            item_index = item_index_old2new[item_index_old]

            user_index_old = int(array[0])

            rating = float(array[2])
            # 按照先前设置的阈值
            if rating >= THRESHOLD[dataset]:
                if user_index_old not in user_pos_ratings:
                    user_pos_ratings[user_index_old] = set()
                user_pos_ratings[user_index_old].add(item_index)
            else:
                if user_index_old not in user_neg_ratings:
                    user_neg_ratings[user_index_old] = set()
                user_neg_ratings[user_index_old].add(item_index)
        except:
            print('the line passed: %s' % line)
            continue
    print('converting rating file ...')
    writer = open('../data/' + dataset + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        # unwatched_set
        unwatched_set = item_set - pos_item_set
        
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]

        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + dataset + '/kg_final.txt', 'w', encoding='utf-8')

    files = []
    if dataset == 'movie1m':
        files.append(open('../data/' + dataset + '/kg_part1_rehashed.txt', encoding='utf-8'))
        files.append(open('../data/' + dataset + '/kg_part2_rehashed.txt', encoding='utf-8'))
    else:
        files.append(open('../data/' + dataset + '/kg_rehashed.txt', encoding='utf-8'))

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)

###########################################
# convert_kg_rehash file function
###########################################
def convert_kg_final_to_kg_rehashed_file():
    return
###########################################
# convert_item_index_to_entity_id function
###########################################
def convert_item_index_to_entity_id(segment):
    file = '../data/' + dataset + '/item_list.txt'
    print('reading item item list file: ' + file + ' ...')
    writer = open('../data/' + dataset + '/item_index2entity_id_rehashed.txt', 'w', encoding='utf-8')
    for line in open(file, encoding='utf-8').readlines():
        try:
            item_index = line.strip().split(segment)[0]
            satori_id = line.strip().split(segment)[1]
            writer.write('{}\t{}\n'.format(item_index, satori_id))
        except:
            print('the line passed: %s' % line)
            continue
    writer.close()

if __name__ == '__main__':
    random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie1m', help='which dataset to preprocess')
    parser.add_argument('-c', '--convert', type=int, default=0, help='if convert item_list to item_index_to_entity_id_file')
    args = parser.parse_args()
    dataset = args.dataset
    is_convert = args.convert
    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    if is_convert:
        convert_item_index_to_entity_id(' ')

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()
    print('[{}]done'.format(dataset))

    DATASETS = ['amazon-book', 'last-fm', 'book', 'movie1m', 'music', 'restaurant', 'movie20m']
    # user_count, item_count, all_entities(in KG and ratings), relations
    number = {}
    # NUMBER = {
    #     'book': {
    #         'users': 17860,
    #         'items': 14966,
    #         'interaction': 139746,
    #         'entities': 77903,
    #         'relations': 25,
    #         'kg_triples': 151500,
    #          'hyperbolicity': 0
    #     },
    #     'movie1m': {
    #         'users': 6036,
    #         'items': 2445,
    #         'interaction': 753772,
    #         'entities': 182011,
    #         'relations': 12,
    #         'kg_triples': 1241995,
    #         'hyperbolicity': 0
    #     },
    #     'music': {
    #         'users': 1872,
    #         'items': 3846,
    #         'interaction': 42346,
    #         'entities': 9366,
    #         'relations': 60,
    #         'kg_triples': 15518,
    #         'hyperbolicity': 0
    #     },
    #     'restaurant': {
    #         'users': 2298698,
    #         'items': 1362,
    #         'interaction': 23416418,
    #         'entities': 28115,
    #         'relations': 7,
    #         'kg_triples': 160519,
    #         'hyperbolicity': 0
    #     },
    #      'movie20m': {
    #         'users': 138159,
    #         'items': 16954,
    #         'interaction': 13501622,
    #         'entities': 102569,
    #         'relations': 32,
    #         'kg_triples': 499474,
    #          'hyperbolicity': 0
    #      }
    # }

    print('DATASETS:' + str(DATASETS))
    non_dataset = []
    for dataset in DATASETS:
        try:
            number[dataset] = get_count(dataset)
        except:
            pass
        if number[dataset] == None:
            number.pop(dataset)
            non_dataset.append(dataset)
    print('NUMBER:' + str(number))
    print('No-dataset: %s' % str(non_dataset))