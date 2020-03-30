########################
# save top@k and draw
########################
import logging
import torch
from eval.topk_eval import topk_eval
from utils import draw_each_mode

########################
# save model
########################
def save_model(args, model, auc):
    if auc:
        model_params_file = f'../model/{args.dataset}_{args.model}_dim{args.dim}_lr.{args.lr}_weight_decay.{args.l2_weight_decay}_params_best_auc.pkl'
        torch.save(model.state_dict(), model_params_file)  # torch.save(model, '../model/gcn_' + DATASET + '.model')
        logging.info(f'Maximum AUC:{auc} Saving successfully.({model_params_file})\n\n')
    else:
        model_params_file = f'../model/{args.dataset}_{args.model}_dim{args.dim}_lr.{args.lr}_weight_decay.{args.l2_weight_decay}_params_earlystop.pkl'
        torch.save(model.state_dict(),
                   model_params_file)  # torch.save(model, '../model/gcn_' + DATASET + '.model')
        logging.info(f'Earlystop. Saving successfully.({model_params_file})\n\n')
########################
# read model
########################
def read_model(model, PATH):
    logging.info(f'Read model:{PATH}')
    model.load_state_dict(torch.load(PATH))
    model.eval()
    logging.info(model.eval())
    return model

def topk(args, device, model, graph, user_list, train_record,test_record, item_set, i_nodes, k_list, auc, is_maxauc=False):
    f1, precision, recall, hr, ndcg = topk_eval(device, model, graph, user_list, train_record,
                                                test_record, item_set, i_nodes, k_list, args.batch_size)
    draw_each_mode(args, [precision, recall, f1, hr, ndcg], auc, '../log/figure/', is_maxauc=is_maxauc)
    if not is_maxauc:
        with open('../log/TOP@K.txt', 'a', encoding='utf-8') as w:
            # 0:music 1:hnn  3:auc 4:Precision 5:Recall 6:f1 7:hr 8:ndcg
            w.write(f'{args.dataset}:{args.model}:0:{auc:.4f}:{precision}:{recall}:{f1}:{hr}:{ndcg}\n')
            # w.write('=========================================================')
            logging.info('TOP@K of the early_stop write into ../log/TOP@K.txt')
    return [precision, recall, f1, hr, ndcg]


def  save_topk(args, auc, topk, is_save):
    # topk: [precision recall f1 hr ndcg]
    if is_save:
        with open('../log/TOP@K.txt', 'a', encoding='utf-8') as w:
            # 0:music 1:hnn  3:auc 4:Precision 5:Recall 6:f1 7:hr 8:ndcg
            w.write(f'{args.dataset}:{args.model}:1:{auc:.4f}:{topk[0]}:{topk[1]}:{topk[2]}:{topk[3]}:{topk[4]}\n')
            # w.write('=========================================================')
            logging.info('TOP@K of the Maximum auc write into ../log/TOP@K.txt')
    else:
        logging.info('Not best auc.')