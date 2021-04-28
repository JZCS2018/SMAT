import logging
import argparse
import math
import os
import sys
import time
import numpy as np
from time import strftime, localtime
import random
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

from data_utils import build_tokenizer, build_embedding_matrix, Dataset

from models import AOA, ABL, SELA, INFERSENT
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        if opt.dataset_file['val']==None:
            fnames = [opt.dataset_file['train'], opt.dataset_file['test']]
        else:
            fnames = [opt.dataset_file['train'], opt.dataset_file['val'], opt.dataset_file['test']]
        tokenizer = build_tokenizer(
            fnames,
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        embedding_matrix = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = Dataset(opt.dataset_file['train'], tokenizer, dat_fname='{0}_train.dat'.format(opt.dataset))
        # self.weight_classes =torch.tensor( compute_class_weight('balanced', np.unique([i['polarity'] for i in self.trainset.data]), self.trainset[4]), dtype = torch.float).to(self.opt.device)
        # self.valset = ABSADataset(opt.dataset_file['val'], tokenizer)self.trainset[4]
        self.testset = Dataset(opt.dataset_file['test'], tokenizer,dat_fname='{0}_test.dat'.format(opt.dataset))
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = Dataset(opt.dataset_file['val'], tokenizer,dat_fname='{0}_val.dat'.format(opt.dataset))

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, lrshrink, minlr):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        epoch = 1
        adam_stop = False
        stop_training = False

        while not stop_training and epoch <= self.opt.num_epoch:
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            ts = time.time()
            for i_batch, sample_batched in enumerate(train_data_loader):
                
                global_step += 1
                # clear gradient accumulators
                self.model.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1, report_val = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, report_val['class1']['f1-score']))
            logger.info('> report: {}'.format(report_val['class1']))
            print('timepassed:')
            logger.info('> timepassed: {}'.format((time.time()-ts)))
            
            epoch +=1

            if report_val['class1']['f1-score'] > max_val_acc:
                max_val_acc = report_val['class1']['f1-score']

                if not os.path.exists('state_dict/'+self.opt.dataset):
                    os.mkdir('state_dict/'+self.opt.dataset)
                path = 'state_dict/'+self.opt.dataset+'/{0}_{1}_val_f1_{2}'.format(self.opt.model_name, self.opt.dataset, round(report_val['class1']['f1-score'], 4))
                

                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            else:
                if 'sgd' in self.opt.optimizer_rec:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / lrshrink
                    logger.info('Shrinking lr by : {0}. New lr = {1}'
                          .format(lrshrink,
                                  optimizer.param_groups[0]['lr']))

                    if optimizer.param_groups[0]['lr'] < minlr:
                        stop_training = True
                if 'adam' in self.opt.optimizer_rec:
                    # early stopping (at 2nd decrease in accuracy)
                    stop_training = adam_stop
                    adam_stop = True
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['class_n'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro')
        report = classification_report(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), target_names=['class0','class1'], output_dict=True, digits = 4)
        # print(report)
        return acc, f1, report

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        if self.opt.valset_ratio==0:
            temp = [t['class_n'] for t in self.trainset.data]
        else:
    	    temp = [t['class_n'] for t in self.trainset.dataset.data]
        class_sample_count = np.array([len(np.where(temp == t)[0]) for t in np.unique(temp)])
        weight_temp = 1. / class_sample_count
        samples_weight = np.array([weight_temp[t] for t in temp])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        if self.opt.valset_ratio==0:
            train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, sampler = sampler)	
            test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
            val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        else:
            train_data_loader = DataLoader(dataset=self.trainset.dataset, batch_size=self.opt.batch_size, sampler = sampler)
            test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
            val_data_loader = DataLoader(dataset=self.valset.dataset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, self.opt.lrshrink, self.opt.minlr)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1 , report_test = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        logger.info('> report: {}'.format(report_test['class1']))
        # print(report_test)
        # logger.info('>> report: {:.4f}'.format(report_test))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='aoa', type=str, help = 'aoa for AOA, abl for Ablation, sela for selfattention')
    parser.add_argument('--dataset', default='webform', type=str, help='cis, cim, ims, cms')
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=8e-1, type=float, help='1e-3')
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=30, type=int, help='default 30')
    parser.add_argument('--batch_size', default=64, type=int, help='default 64')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=240, type=int)
    parser.add_argument('--class_dim', default=2, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=2345, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='default cdm')
    parser.add_argument('--SRD', default=3, type=int, help='default 3')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'aoa': AOA,
        'abl':ABL,
        'sela': SELA,
        'infersent': INFERSENT
    }
    dataset_files = {
    'mimic':{
        'train': './datasets/omap/train_mimic.xlsx',
        'val': './datasets/omap/val_mimic.xlsx',
           'test': './datasets/omap/test_mimic.xlsx'
    },
    'synthea':{
        'train': './datasets/omap/train_synthea.xlsx',
        'val': './datasets/omap/val_synthea.xlsx',
           'test': './datasets/omap/test_synthea.xlsx'
    },
    'cms':{
        'train': './datasets/omap/train_cms.xlsx',
        'val': './datasets/omap/val_cms.xlsx',
           'test': './datasets/omap/test_cms.xlsx'
    },
    'imdb':{
        'train': './datasets/omap/train_imdb.xlsx',
        'val': './datasets/omap/val_imdb.xlsx',
           'test': './datasets/omap/test_imdb.xlsx'
    },
       
    'order':{
        'train': './datasets/omap/train_purchaseorder.xlsx',
        'val': None,
           'test': './datasets/omap/test_purchaseorder.xlsx'
    },
    
    'oaei':{
        'train': './datasets/omap/train_oaei.xlsx',
        # 'val': './datasets/omap/val_oaei.xlsx',
        'val':None,
           'test': './datasets/omap/test_oaei.xlsx'
    },
    'webform':{
        'train': './datasets/omap/train_webform.xlsx',
        'val': './datasets/omap/val_webform.xlsx',
           'test': './datasets/omap/test_webform.xlsx'
    },
    	    

    }
    input_colses = {
        'aoa': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
        'infersent': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
        'sela': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
        'abl': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw':torch.optim.AdamW,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer_rec = opt.optimizer
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if not os.path.exists('logs/'+opt.dataset):
    	os.mkdir('logs/'+opt.dataset)   

    log_file ='./logs/'+opt.dataset +'/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    # log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    # logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
