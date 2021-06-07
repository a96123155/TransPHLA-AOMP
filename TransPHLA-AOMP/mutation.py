from attention import pHLA_attns_draw_save
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from copy import deepcopy
import matplotlib as mpl
import seaborn as sns
import difflib
from collections import OrderedDict

def pHLA_attns_keyaatype_keyaacontrib(data, attns, hla = False, peptide = False): 
    # 已知是负样本的情况下
    pHLA_attns_pd, hla, peptide = pHLA_attns_draw_save(data, attns, hla, peptide)

    pHLA_keyaatype = OrderedDict()
    for posi, pep_aa, aa_attn_sum in zip(list(pHLA_attns_pd.loc['posi']),
                                         list(pHLA_attns_pd.columns),
                                         list(pHLA_attns_pd.loc['sum'])):

        pHLA_keyaatype[int(posi)] = [pep_aa, aa_attn_sum]
    pHLA_keyaatype = OrderedDict(sorted(pHLA_keyaatype.items(), key = lambda t: (-t[1][1])))

    pHLA_keyaatype_contrib = OrderedDict()
    for posi, pep_aa, aa_attn_contrib in zip(list(pHLA_attns_pd.loc['posi']),
                                         list(pHLA_attns_pd.columns),
                                         list(pHLA_attns_pd.loc['contrib'])):

        pHLA_keyaatype_contrib[int(posi)] = [pep_aa, aa_attn_contrib]
    pHLA_keyaatype_contrib = OrderedDict(sorted(pHLA_keyaatype_contrib.items(), key = lambda t: (-t[1][1])))

    return pHLA_attns_pd, pHLA_keyaatype, pHLA_keyaatype_contrib

def HLA_length_aatype_position_num(hla = False, length = 9, label = None):

    if label == 'None': 
        new_label = 'all'
    elif label == 1:
        new_label = 'positive'
    elif label == 0:
        new_label == 'negative'
    
    try:
        aatype_position = np.load('./Attention/peptideAAtype_peptidePosition/{}_Length{}.npy'.format(hla, length),
                                  allow_pickle = True).item()[new_label]

        aatype_position_num = np.load('./Attention/peptideAAtype_peptidePosition_NUM/{}_Length{}_num.npy'.format(hla, length),
                                       allow_pickle = True).item()[new_label]
    except:
        print('No {} with {}, Use the overall attention for pepAAtype-peppsition'.format(hla, length))
        aatype_position = np.load('./Attention/peptideAAtype_peptidePosition/Allsamples_Alllengths.npy',
                                  allow_pickle = True).item()[length][new_label]
        aatype_position_num = np.load('./Attention/peptideAAtype_peptidePosition_NUM/Allsamples_Alllengths_num.npy',
                                      allow_pickle = True).item()[length][new_label]
    
    aatype_position.loc['sum'] = aatype_position.sum(axis = 0)
    aatype_position['sum'] = aatype_position.sum(axis = 1)
    
    return aatype_position, aatype_position_num

def HLA_aatype_position_contribution(aatype_position_pd, aatype_position_num_pd, length = 9):
    contrib = np.zeros((20, length))
    for aai, aa in enumerate(aatype_position_pd.index[:-1]): # 有个sum
        for pi, posi in enumerate(aatype_position_pd.columns[:-1]): # 有个sum
            
            p_aa_posi = aatype_position_pd.loc[aa, posi] / aatype_position_num_pd.loc[aa, posi]
            p_posi = aatype_position_num_pd.loc['sum', 'sum'] / aatype_position_pd.loc['sum', 'sum']
            
            contrib[aai, pi] = p_aa_posi * p_posi
            
    contrib = pd.DataFrame(contrib, index = aatype_position_pd.index[:-1], columns = aatype_position_pd.columns[:-1])
    contrib.fillna(0, inplace = True)
    return contrib

def HLA_Length_Label_pepaatype_pepposition(hla = False, length = 9, label = 1):
    aatype_position_pd, aatype_position_num_pd = HLA_length_aatype_position_num(hla, length, label)
    aatype_position_contrib_pd = HLA_aatype_position_contribution(aatype_position_pd, aatype_position_num_pd, length)
    return aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd

def HLA_Length_position_keyaatype(aatype_position_pd, aatype_position_contrib_pd, length = 9):
    
    position_contrib_keyaatype = OrderedDict()
    for posi in range(1, length + 1):
        temp_sorted = aatype_position_contrib_pd[posi].sort_values(ascending = False)
        key_aatype = [k for k,v in OrderedDict(temp_sorted > 1).items() if v]
        position_contrib_keyaatype[posi] = [key_aatype, len(key_aatype), temp_sorted.max().round(2), temp_sorted.mean().round(2)]
    position_contrib_keyaatype = OrderedDict(sorted(position_contrib_keyaatype.items(), key = lambda t: (-t[1][2], t[1][1], -t[1][3])))
   
    ###################
    
    aatype_position_pd = aatype_position_pd.drop(index = 'sum')
    aatype_position_pd = aatype_position_pd.drop(columns = 'sum')
    position_keyaatype = OrderedDict()
    for posi in range(1, length + 1):
        temp_sorted = aatype_position_pd[posi].sort_values(ascending = False)
        key_aatype = [k for k,v in OrderedDict(temp_sorted > aatype_position_pd[posi].mean()).items() if v]
        position_keyaatype[posi] = [key_aatype, len(key_aatype), temp_sorted.max().round(2), temp_sorted.mean().round(2)]
    position_keyaatype = OrderedDict(sorted(position_keyaatype.items(), key = lambda t: (t[1][1], -t[1][2], t[1][3])))
    
    return position_contrib_keyaatype, position_keyaatype

def HLA_Length_aatype_position_contrib_attn_num(hla = False, length = 9, label = 1):
    aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd = \
    HLA_Length_Label_pepaatype_pepposition(hla, length, label)

    position_contrib_keyaatype, position_keyaatype = \
    HLA_Length_position_keyaatype(aatype_position_pd, aatype_position_contrib_pd, length)
    
    return position_contrib_keyaatype, position_keyaatype, aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd

def HLA_pHLA_contrib_keyaatype_attn_num(data, attn_data, hla = False, peptide = False, label = 1):
    
    length = len(peptide)

    ### HLA positive
    position_contrib_keyaatype, position_keyaatype, aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd\
    = HLA_Length_aatype_position_contrib_attn_num(hla, length, label)

    ### pHLA
    pHLA_attns_pd, pHLA_keyaatype, pHLA_keyaatype_contrib = pHLA_attns_keyaatype_keyaacontrib(data, attn_data, hla, peptide)
    
    return position_contrib_keyaatype, position_keyaatype, aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd, \
            pHLA_attns_pd, pHLA_keyaatype, pHLA_keyaatype_contrib
    
##############################################################################
def oneposition_mut_peptides(mut_posi, mut_aatypes, peptide, print_ = False):

    mut_peptides = []
    for mut_aa in mut_aatypes:
        index = mut_posi - 1
        mut_peptide = peptide[:index] + mut_aa + peptide[index+1:]
        mut_peptides.append(mut_peptide)
    if print_: print(mut_peptides)
    return mut_peptides

def find_mutate_position_aatype(hla_oripeptide_all_mutpeptides):
    original_peptide = hla_oripeptide_all_mutpeptides[0]
    pep_length = len(original_peptide)
    mutate_position_aatype, mutate_num = [], []
    for pep in hla_oripeptide_all_mutpeptides:
        s = ''
        for i in range(pep_length):
            if original_peptide[i] != pep[i]:
                s += str(i+1) + '|' + original_peptide[i] + '/' + pep[i] + ','
        mutate_num.append(len(s[:-1].split(',')))
        mutate_position_aatype.append(s[:-1])
    return mutate_position_aatype, mutate_num

def mutation_positive_peptide_1(hla_peptide_keyaatype_contrib,
                              HLA_length_positive_position_contrib_keyaatype,
                              HLA_length_positive_position_keyaatype,
                              peptide = 'SRELSFTSA', hla = 'HLA-A*68:01', print_ = False):
    # 用贡献值排序
    neg_mut_position_order = list(hla_peptide_keyaatype_contrib.keys())
    pos_mut_position_order = list(HLA_length_positive_position_contrib_keyaatype.keys())
    if print_: print(neg_mut_position_order, pos_mut_position_order)
    # 用贡献值对比，真实值排序
    mut_peptides_step = dict()
    mut_peptides_step[0] = peptide
    all_peptides = []
    
    i, pos_i, neg_i = 0, 0, 0
    mut_positions = []
    while i <= 3:
        
        for idx, item in enumerate(neg_mut_position_order):
            if item not in mut_positions:
                posi = item
                neg_i = idx
                break
        for idx, item in enumerate(pos_mut_position_order):
            if item not in mut_positions:
                pos_i = idx
                break
            
        (aatype, attnsum) = hla_peptide_keyaatype_contrib[posi]
        if attnsum < HLA_length_positive_position_contrib_keyaatype[pos_mut_position_order[pos_i]][3]: # 比较contrib的mean
            mut_posi = pos_mut_position_order[pos_i]
            mut_aatypes_all = HLA_length_positive_position_keyaatype[mut_posi][0] # 用真实值的aatype
            mut_aatypes = [aa for aai, aa in enumerate(mut_aatypes_all) if aai < 3]
            if print_: print('Effect of {}{} in Negative {} < Effect of {}{} in Positive {}, Replace to {}{}'.format(posi, aatype, peptide,mut_posi, mut_aatypes_all, hla, mut_posi, mut_aatypes))
            
        else:
            mut_posi = posi
            mut_aatypes_all = HLA_length_positive_position_keyaatype[mut_posi][0]
            mut_aatypes = [aa for aai, aa in enumerate(mut_aatypes_all) if aai < 3]
            if print_: print('Effect of {}{} in Negative {} > Effect of {}{} in Positive {}, Replace to {}{}'.format(posi, aatype, peptide, mut_posi, mut_aatypes_all, hla, mut_posi, mut_aatypes))
            
        if i == 0: 
            mut_peptides = oneposition_mut_peptides(mut_posi, mut_aatypes, peptide, print_)
        else:
            mut_peptides_new = []
            for pep in mut_peptides:
                mut_peptides_new.extend(oneposition_mut_peptides(mut_posi, mut_aatypes, pep, print_))
            mut_peptides = mut_peptides_new
        mut_peptides_step[i+1] = mut_peptides
        all_peptides.extend(mut_peptides)
         
        mut_positions.append(mut_posi)
        i += 1 

    return mut_peptides_step, all_peptides

def mutation_positive_peptide_2(hla_peptide_keyaatype,
                              HLA_length_positive_position_keyaatype,
                              peptide = 'SRELSFTSA', hla = 'HLA-A*68:01', print_ = False):
    # 用真实值排序
    neg_mut_position_order = list(hla_peptide_keyaatype.keys())
    pos_mut_position_order = list(HLA_length_positive_position_keyaatype.keys())
    if print_: print(neg_mut_position_order, pos_mut_position_order)
    # 用真实值对比：优先考虑正样本贡献（正样本是所有样本的累加值，所以可以认为一定大于负样本）
    mut_peptides_step = dict()
    mut_peptides_step[0] = peptide
    all_peptides = []
    
    i, pos_i, neg_i = 0, 0, 0
    mut_positions = []
    while i <= 3:
        
        for idx, item in enumerate(neg_mut_position_order):
            if item not in mut_positions:
                posi = item
                neg_i = idx
                break
        for idx, item in enumerate(pos_mut_position_order):
            if item not in mut_positions:
                pos_i = idx
                break
        
        (aatype, attnsum) = hla_peptide_keyaatype[posi]
        if attnsum < HLA_length_positive_position_keyaatype[pos_mut_position_order[pos_i]][3]: # 比较contrib的mean
            mut_posi = pos_mut_position_order[pos_i]
            mut_aatypes_all = HLA_length_positive_position_keyaatype[mut_posi][0] # 用真实值的aatype
            mut_aatypes = [aa for aai, aa in enumerate(mut_aatypes_all) if aai < 3]
            if print_: print('Effect of {}{} in Negative {} < Effect of {}{} in Positive {}, Replace to {}{}'.format(posi, aatype, peptide,mut_posi, mut_aatypes_all, hla, mut_posi, mut_aatypes))

        else:
            mut_posi = posi
            mut_aatypes_all = HLA_length_positive_position_keyaatype[mut_posi][0]
            mut_aatypes = [aa for aai, aa in enumerate(mut_aatypes_all) if aai < 3]
            if print_: print('Effect of {}{} in Negative {} > Effect of {}{} in Positive {}, Replace to {}{}'.format(posi, aatype, peptide, mut_posi, mut_aatypes_all, hla, mut_posi, mut_aatypes))

        if i == 0: 
            mut_peptides = oneposition_mut_peptides(mut_posi, mut_aatypes, peptide, print_)
        else:
            mut_peptides_new = []
            for pep in mut_peptides:
                mut_peptides_new.extend(oneposition_mut_peptides(mut_posi, mut_aatypes, pep, print_))
            mut_peptides = mut_peptides_new
        mut_peptides_step[i+1] = mut_peptides
        all_peptides.extend(mut_peptides)
         
        mut_positions.append(mut_posi)
        i += 1 

    return mut_peptides_step, all_peptides

def mutation_positive_peptide_3(hla_peptide_keyaatype,
                              HLA_length_positive_position_keyaatype,
                              peptide = 'SRELSFTSA', hla = 'HLA-A*68:01', print_ = False):
    # 用真实值排序
    neg_mut_position_order = list(hla_peptide_keyaatype.keys())
    pos_mut_position_order = list(HLA_length_positive_position_keyaatype.keys())
    if print_: print(neg_mut_position_order, pos_mut_position_order)
    # 用真实值对比：优先考虑负样本贡献（正样本是所有样本的累加值，所以可以认为一定大于负样本）
    mut_peptides_step = dict()
    mut_peptides_step[0] = peptide
    all_peptides = []
    
    i, pos_i, neg_i = 0, 0, 0
    mut_positions = []
    while i <= 3:
        for idx, item in enumerate(neg_mut_position_order):
            if item not in mut_positions:
                posi = item
                neg_i = idx
                break
        for idx, item in enumerate(pos_mut_position_order):
            if item not in mut_positions:
                pos_i = idx
                break
        
        (aatype, attnsum) = hla_peptide_keyaatype[posi]
        if i == 0 or attnsum > HLA_length_positive_position_keyaatype[pos_mut_position_order[i]][3]: # 比较contrib的mean
            mut_posi = posi
            mut_aatypes_all = HLA_length_positive_position_keyaatype[mut_posi][0]
            mut_aatypes = [aa for aai, aa in enumerate(mut_aatypes_all) if aai < 3]
            if print_: print('Effect of {}{} in Negative {} > Effect of {}{} in Positive {}, Replace to {}{}'.format(posi, aatype, peptide, mut_posi, mut_aatypes_all, hla, mut_posi, mut_aatypes))
        
        else:
            mut_posi = pos_mut_position_order[pos_i]
            mut_aatypes_all = HLA_length_positive_position_keyaatype[mut_posi][0] # 用真实值的aatype
            mut_aatypes = [aa for aai, aa in enumerate(mut_aatypes_all) if aai < 3]
            if print_: print('Effect of {}{} in Negative {} < Effect of {}{} in Positive {}, Replace to {}{}'.format(posi, aatype, peptide,mut_posi, mut_aatypes_all, hla, mut_posi, mut_aatypes))
             
        if i == 0: 
            mut_peptides = oneposition_mut_peptides(mut_posi, mut_aatypes, peptide, print_)
        else:
            mut_peptides_new = []
            for pep in mut_peptides:
                mut_peptides_new.extend(oneposition_mut_peptides(mut_posi, mut_aatypes, pep, print_))
            mut_peptides = mut_peptides_new
        mut_peptides_step[i+1] = mut_peptides
        all_peptides.extend(mut_peptides)
        
        mut_positions.append(mut_posi)
        i += 1        

    return mut_peptides_step, all_peptides

def mutation_positive_peptide_4(hla_peptide_keyaatype,
                              HLA_length_positive_position_keyaatype,
                              peptide = 'SRELSFTSA', hla = 'HLA-A*68:01', print_ = False):
    # 用真实值排序
    neg_mut_position_order = list(hla_peptide_keyaatype.keys())
    if len(peptide) == 8:
        pos_mut_position_order = [i for i in list(HLA_length_positive_position_keyaatype.keys()) if i in [1, 2, 8]]
    else:
        pos_mut_position_order = [i for i in list(HLA_length_positive_position_keyaatype.keys()) if i in [1, 2, 9]]
    if print_: print(neg_mut_position_order, pos_mut_position_order)
    # 改变129位+neg位
    mut_peptides_step = dict()
    mut_peptides_step[0] = peptide
    all_peptides = []
    pos_i, neg_i = -1, -1
    mut_positions = []
    i = 0
    while len(mut_positions) < 4:
        
        if i < 3: # 0 1 2
            mut_posi = pos_mut_position_order[i]
        elif i == 3:
            for item in neg_mut_position_order:
                if item not in mut_positions:
                    mut_posi = item
                    break
            
        (aatype, attnsum) = hla_peptide_keyaatype[mut_posi]
        mut_aatypes_all = HLA_length_positive_position_keyaatype[mut_posi][0] # 用真实值的aatype
        mut_aatypes = [aa for aai, aa in enumerate(mut_aatypes_all) if aai < 3]
        if print_: print('{}{} in Negative {} → {}{} in Positive {}'.format(mut_posi, aatype, peptide,mut_posi, mut_aatypes, hla))
            
        if i == 0: 
            mut_peptides = oneposition_mut_peptides(mut_posi, mut_aatypes, peptide, print_)
        else:
            mut_peptides_new = []
            for pep in mut_peptides:
                mut_peptides_new.extend(oneposition_mut_peptides(mut_posi, mut_aatypes, pep, print_))
            mut_peptides = mut_peptides_new
        mut_peptides_step[i+1] = mut_peptides
        all_peptides.extend(mut_peptides)
        
        mut_positions.append(mut_posi)
        i += 1
              
    return mut_peptides_step, all_peptides

def pHLA_mutation_peptides(data, attn_data, idx = -1, hla = False, peptide = False, print_ = False):
    if not(peptide and hla) and idx > -1:
        hla = data.iloc[idx].HLA
        peptide = data.iloc[idx].peptide
        
    HLA_Length_Positive_position_contrib_keyaatype, HLA_Length_Positive_position_keyaatype, \
    HLA_Length_Positive_aatype_position_contrib_pd, HLA_Length_Positive_aatype_position_pd, \
    HLA_Length_Positive_aatype_position_num_pd, \
    pHLA_attns_pd, pHLA_keyaatype, pHLA_keyaatype_contrib = HLA_pHLA_contrib_keyaatype_attn_num(data, 
                                                                     attn_data, 
                                                                     hla, 
                                                                     peptide, 
                                                                     label = 1)

    mut_peptides = []
    if print_: print('********** Strategy 1 **********')
    pHLA_mut_peptides_step_1, pHLA_mut_peptides_1 = \
    mutation_positive_peptide_1(
        pHLA_keyaatype_contrib,
        HLA_Length_Positive_position_contrib_keyaatype,
        HLA_Length_Positive_position_keyaatype,
        peptide = peptide, hla = hla, print_ = print_)
    if print_: print('********** Strategy 2 **********')
    pHLA_mut_peptides_step_2, pHLA_mut_peptides_2 = \
    mutation_positive_peptide_2(
        pHLA_keyaatype,
        HLA_Length_Positive_position_keyaatype,
        peptide = peptide, hla = hla, print_ = print_)
    if print_: print('********** Strategy 3 **********')
    pHLA_mut_peptides_step_3, pHLA_mut_peptides_3 = \
    mutation_positive_peptide_3(
        pHLA_keyaatype,
        HLA_Length_Positive_position_keyaatype,
        peptide = peptide, hla = hla, print_ = print_)
    if print_: print('********** Strategy 4 **********')
    pHLA_mut_peptides_step_4, pHLA_mut_peptides_4 = \
    mutation_positive_peptide_4(
        pHLA_keyaatype,
        HLA_Length_Positive_position_keyaatype,
        peptide = peptide, hla = hla, print_ = print_)

    mut_peptides.extend(pHLA_mut_peptides_1)
    mut_peptides.extend(pHLA_mut_peptides_2)
    mut_peptides.extend(pHLA_mut_peptides_3)
    mut_peptides.extend(pHLA_mut_peptides_4)

    mut_peptides = list(set(mut_peptides))
    mut_peptides = [peptide] + mut_peptides

    mutate_position_aatype = find_mutate_position_aatype(mut_peptides)

    all_peptides_df = pd.DataFrame([[peptide] * len(mut_peptides),
                                    mut_peptides, 
                                    mutate_position_aatype[0],
                                    mutate_position_aatype[1],
                                    [difflib.SequenceMatcher(None, item ,peptide).ratio() for item in mut_peptides]],
        index = ['original_peptide', 'mutation_peptide', 'mutation_position_AAtype', 'mutation_AA_number', 'sequence similarity']).T.drop_duplicates().sort_values(by = 'mutation_AA_number').reset_index(drop = True)
    all_peptides_df['HLA'] = hla
    return all_peptides_df # ' '.join(all_peptides_df.mutation_peptide), 