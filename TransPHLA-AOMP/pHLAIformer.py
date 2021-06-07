
# coding: utf-8


from model import *
from attention import pHLA_attns_draw_save
from mutation import *
from utils import Logger, cut_peptide_to_specific_length

import math
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(1234)
from scipy import interp
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from collections import OrderedDict
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns

import difflib

seed = 19961231
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import os
import argparse
import logging
import sys

parser = argparse.ArgumentParser(usage = 'peptide-HLA-I binding prediction')
parser.add_argument('--peptide_file', type = str, help = 'the path of the .fasta file contains peptides')
parser.add_argument('--HLA_file', type = str, help = 'the path of the .fasta file contains sequence')
parser.add_argument('--threshold', type = float, default = 0.5, help = 'the threshold to define predicted binder, float from 0 - 1, the recommended value is 0.5')
parser.add_argument('--cut_peptide', type = bool, default = True, help = 'Whether to split peptides larger than cut_length?')
parser.add_argument('--cut_length', type = int, default = 9, help = 'if there is a peptide sequence length > 15, we will segment the peptide according the length you choose, from 8 - 15')
parser.add_argument('--output_dir', type = str, help = 'The directory where the output results are stored.')
parser.add_argument('--output_attention', type = bool, default = True, help = 'Output the mutual influence of peptide and HLA on the binding?')
parser.add_argument('--output_heatmap', type = bool, default = True, help = 'Visualize the mutual influence of peptide and HLA on the binding?')
parser.add_argument('--output_mutation', type = bool, default = True, help = 'Whether to perform mutations with better affinity for each sample?')
args = parser.parse_args()
print(args)

errLogPath = args.output_dir + '/error.log'
if args.threshold <= 0 or args.threshold >= 1: 
    log = Logger(errLogPath)
    log.logger.critical('The threshold invalid, please check whether it ranges from 0-1.')
    sys.exit(0)
if not args.peptide_file:
    log = Logger(errLogPath)
    log.logger.critical('The peptide file is empty.')
    sys.exit(0)
if not args.HLA_file:
    log = Logger(errLogPath)
    log.logger.critical('The HLA file is empty.')
    sys.exit(0)
if not args.output_dir:
    log = Logger(errLogPath)
    log.logger.critical('Please fill the output file directory.')
    sys.exit(0)
	
cut_length = args.cut_length
if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

# # 读取文件

with open(args.peptide_file, 'r') as f:
    peptide_file = f.readlines()

with open(args.HLA_file, 'r') as f:
    HLA_file = f.readlines()
	
if len(peptide_file) != len(HLA_file):
	log = Logger(errLogPath)
    log.logger.critical('Please ensure the same number of HLAs and peptides.')
    sys.exit(0)
	
i = 0
ori_peptides, ori_HLA_names, ori_HLA_sequences = [], [], []
for pep, hla in zip(peptide_file, HLA_file):
    if i % 2 == 0:
        hla_name = hla.replace('>', '').replace('\t', '').replace('\n', '')
        ori_HLA_names.append(hla_name)
    if i % 2 == 1:
        hla_seq = str.upper(hla.replace('\n', '').replace('\t', ''))
        peptide = str.upper(pep.replace('\n', '').replace('\t', ''))
        ori_peptides.append(peptide)
        ori_HLA_sequences.append(hla_seq)
    i += 1

peptides, HLA_names, HLA_sequences = [], [], []
for pep, hla_name, hla_seq in zip(ori_peptides, ori_HLA_names, ori_HLA_sequences):
    
    if not (pep.isalpha() and hla.isalpha()): 
        continue
    if len(set(pep).difference(set('ARNDCQEGHILKMFPSTWYV'))) != 0:
        continue
    if len(set(hla_seq).difference(set('ARNDCQEGHILKMFPSTWYV'))) != 0:
        continue
	if len(hla_seq) > 34:
		continue
    length = len(pep)
    if length < 15:
        if args.cut_peptide:
            if length > cut_length:
                cut_peptides = [pep] + [pep[i : i + cut_length] for i in range(length - cut_length + 1)]
                peptides.extend(cut_peptides)
                HLA_sequences.extend([hla_seq] * len(cut_peptides))
                HLA_names.extend([hla_name] * len(cut_peptides))
            else:
                peptides.append(pep)
                HLA_sequences.append(hla_seq)
                HLA_names.append(hla_name)
        else:
            peptides.append(pep)
            HLA_sequences.append(hla_seq)
            HLA_names.append(hla_name)
            
    else:
        cut_peptides = [pep[i : i + cut_length] for i in range(length - cut_length + 1)]
        peptides.extend(cut_peptides)
        HLA_sequences.extend([hla_seq] * len(cut_peptides))
        HLA_names.extend([hla_name] * len(cut_peptides))
        
predict_data = pd.DataFrame([HLA_names, HLA_sequences, peptides], index = ['HLA', 'HLA_sequence', 'peptide']).T
if predict_data.shape[0] == 0: 
    log = Logger(errLogPath)
    log.logger.critical('No suitable data could be predicted. Please check your input data.')
    sys.exit(0)
    
predict_data, predict_pep_inputs, predict_hla_inputs, predict_loader = read_predict_data(predict_data, batch_size)

# # 预测

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

model_file = 'pHLAIformer.pkl'

model_eval = Transformer().to(device)
model_eval.load_state_dict(torch.load(model_file, map_location='cpu'), strict = True)

model_eval.eval()
y_pred, y_prob, attns = eval_step(model_eval, predict_loader, args.threshold, use_cuda)

predict_data['y_pred'], predict_data['y_prob'] = y_pred, y_prob
predict_data = predict_data.round({'y_prob': 4})

predict_data.to_csv(args.output_dir + '/predict_results.csv', index = False)

# # 作图

if args.output_attention or args.output_heatmap:
    
    if args.output_attention: 
        attn_savepath = args.output_dir + '/attention/'
        if not os.path.exists(attn_savepath):
            os.makedirs(attn_savepath)
    else:
        attn_savepath = False
    if args.output_heatmap: 
        fig_savepath = args.output_dir + '/figures/'
        if not os.path.exists(fig_savepath):
            os.makedirs(fig_savepath)
    else:
        fig_savepath = False
        
    for hla, pep in zip(predict_data.HLA, predict_data.peptide):
        pHLA_attns_draw_save(predict_data, attns, hla, pep, attn_savepath, fig_savepath)


# # 突变

if args.output_mutation:
    
    mut_savepath = args.output_dir + '/mutation/'
    if not os.path.exists(mut_savepath):
        os.makedirs(mut_savepath)
    
    for idx in range(predict_data.shape[0]):
        peptide = predict_data.iloc[idx].peptide
        hla = predict_data.iloc[idx].HLA
        
        if len(peptide) < 8 or len(peptide) > 14: continue
            
        mut_peptides_df = pHLA_mutation_peptides(predict_data, attns, hla = hla, peptide = peptide)
        mut_data, _, _, mut_loader = read_predict_data(mut_peptides_df, batch_size)

        model_eval = Transformer().to(device)
        model_eval.load_state_dict(torch.load(model_file, map_location='cpu'), strict = True)

        model_eval.eval()
        y_pred, y_prob, attns = eval_step(model_eval, mut_loader, args.threshold, use_cuda)

        mut_data['y_pred'], mut_data['y_prob'] = y_pred, y_prob
        mut_data = mut_data.round({'y_prob': 4})
        mut_data.to_csv(mut_savepath + '{}_{}_mutation.csv'.format(hla, peptide), index = False)
        print('********** {} | {} → # Mutation peptides = {}'.format(hla, peptide, mut_data.shape[0]-1))
        
        mut_peptides_IEDBfmt = ' '.join(mut_data.mutation_peptide)
        print('If you want to use IEDB tools to predict IC50, please use these format: \n {}'.format(mut_peptides_IEDBfmt))

