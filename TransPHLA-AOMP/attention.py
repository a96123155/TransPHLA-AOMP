import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from copy import deepcopy
import matplotlib as mpl
import seaborn as sns
import difflib

def sort_aatype(df):
    aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
    df.reset_index(inplace = True)
    df['index'] = df['index'].astype('category')
    df['index'].cat.reorder_categories(aatype_sorts, inplace=True)
    df.sort_values('index', inplace=True)
    df.rename(columns = {'index':''}, inplace = True)
    df = df.set_index('')
    return df

# Peptide length-Peptide position
def attn_sumhead_peplength_pepposition(data, attn_data, label = None):
    SUM_length_head_dict = {}
    for l in range(8, 15):
        print('Length = ', str(l))
        SUM_length_head_dict[l] = []
        
        if label == None:
            length_index = np.array(data[data.length == l].index)
        elif label == 1:
            length_index = np.array(data[data.label == 1][data.length == l].index)
        elif label == 0:
            length_index = np.array(data[data.label == 0][data.length == l].index)
            
        length_data_num = len(length_index)
        print(length_data_num, length_index)

        for head in range(9):
            idx_0 = length_index[0]
            temp_length_head = deepcopy(nn.Softmax(dim = -1)(attn_data[idx_0][head][:, :l].float())) # Shape = (34, length), 行是HLA，列是peptide，由行查列

            for idx in length_index[1:]:
                temp_length_head += nn.Softmax(dim = -1)(attn_data[idx][head][:, :l].float())

            temp_length_head = np.array(nn.Softmax(dim = -1)(temp_length_head.sum(axis = 0))) # 把这一列的数据相加，shape = （length，）
            SUM_length_head_dict[l].append(temp_length_head)
            
    #############################
    SUM_length_head_sum = []
    for l in range(8, 15):
        temp = pd.DataFrame(SUM_length_head_dict[l], columns = range(1, l+1)).round(4)
        temp.loc['sum'] = temp.sum(axis = 0)
        SUM_length_head_sum.append(list(temp.loc['sum']))
        print(l, temp.loc['sum'].sort_values(ascending = False).index)
        
    return SUM_length_head_dict, SUM_length_head_sum

# 对于每一种peptide length：Head-peptide position
def draw_eachlength_head_pepposition(sum_peplength_pepposition, label, savepath = False):
    
    fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (15, 6), dpi = 600)
    fig.patch.set_facecolor('white')
    cmap = 'cividis'
    
    for l in range(8, 15):
        temp = pd.DataFrame(sum_peplength_pepposition[l], columns = range(1, l+1)).round(4)
        sns.heatmap(temp, cmap = cmap, cbar = False, square = True, ax = axes[(l-8)//4, (l-8)%4], 
                    xticklabels = range(1,l+1), yticklabels = range(1, 10))
        axes[(l-8)//4, (l-8)%4].set_title('Peptide Length = {}'.format(l))
        
    axes[0, 0].set_ylabel('Head')
    axes[1, 0].set_ylabel('Head')
    axes[0, 3].set_xlabel('Peptide position')
    axes[1, 0].set_xlabel('Peptide position')
    axes[1, 1].set_xlabel('Peptide position')
    axes[1, 2].set_xlabel('Peptide position')
    axes[1, 3].axis('off')
    
    if label == None:
        label = 'All'
    else:
        label = ['Negative', 'Positive'][label == 1]
    fig.suptitle('{} samples | Heads - Peptide positions'.format(label), x = 0.43)

    norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), 
                        orientation='vertical', ax = axes, shrink = 1, anchor = (-0.2, 0.5))
    cbar.outline.set_visible(False)
    
    if savepath:
        plt.savefig('./figures/pHLAIformer/Attention/{} samples_eachPepLength__Heads_PepPositions.tif'.format(label), dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())

    plt.show()
    

# 针对每一个head：Peptide length - Peptide position
def draw_eachhead_peplength_pepposition(SUM_length_head_dict, label = None, savepath = False):
    
    SUM_head_length_dict = dict()
    for l in range(8, 15):
        for head in range(9):
            SUM_head_length_dict.setdefault(head, [])
            SUM_head_length_dict[head].append(SUM_length_head_dict[l][head])
    assert len(SUM_head_length_dict[1]) == 7
    assert len(SUM_head_length_dict.keys()) == 9

    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (15, 7), dpi = 600)
    fig.patch.set_facecolor('white')
    cmap = 'cividis'
    
    for head in range(9):
        temp = pd.DataFrame(SUM_head_length_dict[head], columns = range(1, 15), index = range(8, 15))
#         cbar = [False, True][head == 2]
        sns.heatmap(temp, cmap = cmap, cbar = False, square = True, xticklabels = True, yticklabels = True, ax = axes[head//3, head%3])
        axes[head//3, head%3].set_title('Head '+ str(head))
    axes[0, 0].set_ylabel('Peptide length')
    axes[1, 0].set_ylabel('Peptide length')
    axes[2, 0].set_ylabel('Peptide length')
    axes[2, 0].set_xlabel('Peptide position')
    axes[2, 1].set_xlabel('Peptide position')
    axes[2, 2].set_xlabel('Peptide position') 
    
    if label == None:
        label = 'All'
    else:
        label = ['Negative', 'Positive'][label == 1]
    fig.suptitle('{} samples | Peptide lengths - Peptide positions'.format(label), x = 0.42)
    
    norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), 
                        orientation='vertical', ax = axes, shrink = 1, anchor = (-0.2, 0.5))
    cbar.outline.set_visible(False)
    
    if savepath:
        plt.savefig('./figures/pHLAIformer/Attention/{} samples_eachHead__PepLengths_PepPositions.tif'.format(label), dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())

    plt.show()

# 不分length一起计算: Heads - PepPositions
def attn_SUMlength_heads_peppositions(data, attn_data, label = None):
    SUMlength_heads_peppositions = []
    if label == None:
        index = np.array(data.index)
    else:
        index = np.array(data[data.label == label].index)

    for head in range(9):
        idx_0 = index[0]
        temp_length_head = deepcopy(nn.Softmax(dim=-1)(attn_data[idx_0][head][:, :14].float())) # Shape = (34, 14), 行是HLA，列是peptide，由行查列

        for idx in index[1:]:
            temp_length_head += nn.Softmax(dim=-1)(attn_data[idx][head][:, :14].float())

        temp_length_head = np.array(nn.Softmax(dim=-1)(temp_length_head.sum(axis = 0))) # 把这一列的数据相加，shape = （14，）
        SUMlength_heads_peppositions.append(temp_length_head)
    return SUMlength_heads_peppositions
    
# peptide AA type - peptide position
def attn_HLA_length_aatype_position_num(data, attn_data, hla = 'HLA-A*11:01', label = None, length = 9, show_num = False):
    aatype_position = dict()
    if label == None:
        length_index = np.array(data[data.length == length][data.HLA == hla].index)
    else:
        length_index = np.array(data[data.length == length][data.HLA == hla][data.label == label].index)

    length_data_num = len(length_index)

    for head in range(9):
        for idx in length_index:
            temp_peptide = data.iloc[idx].peptide
            temp_length_head = deepcopy(nn.Softmax(dim=-1)(attn_data[idx][head][:, :length].float())) # Shape = (34, 9), 行是HLA，列是peptide，由行查列
            temp_length_head = nn.Softmax(dim=-1)(temp_length_head.sum(axis = 0)) # 把这一列的数据相加，shape = （9，）

            for i, aa in enumerate(temp_peptide): 
                aatype_position.setdefault(aa, {})
                aatype_position[aa].setdefault(i, 0)
                aatype_position[aa][i] += temp_length_head[i] 
    
    if show_num:
        aatype_position_num = dict()
        for idx in length_index:
            temp_peptide = data.iloc[idx].peptide
            for i, aa in enumerate(temp_peptide):
                aatype_position_num.setdefault(aa, {})
                aatype_position_num[aa].setdefault(i, 0)
                aatype_position_num[aa][i] += 1
             
        return aatype_position, aatype_position_num
    else:
        return aatype_position
    
def attn_HLA_length_aatype_position_pd(HLA_length_aatype_position, length = 9, softmax = True, unsoftmax = True):
        
    HLA_length_aatype_position_pd = np.zeros((20, length))
    
    aai, aa_indexs = 0, []
    for aa, aa_posi in HLA_length_aatype_position.items():
        aa_indexs.append(aa)
        for posi, v in aa_posi.items():
            HLA_length_aatype_position_pd[aai, posi] = v
        aai += 1
    
    if len(aa_indexs) != 20: 
        aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
        abscent_aa = list(set(aatype_sorts).difference(set(aa_indexs)))
        aa_indexs += abscent_aa
    
    if softmax and not unsoftmax: 
        HLA_length_aatype_position_softmax_pd = deepcopy(nn.Softmax(dim = -1)(torch.Tensor(HLA_length_aatype_position_pd)))
        HLA_length_aatype_position_softmax_pd = np.array(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, length + 1))
        return HLA_length_aatype_position_softmax_pd
    
    elif unsoftmax and not softmax:
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd,
                                                               index = aa_indexs, columns = range(1, length + 1))
        return HLA_length_aatype_position_unsoftmax_pd
    
    elif softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd = deepcopy(nn.Softmax(dim = -1)(torch.Tensor(HLA_length_aatype_position_pd)))
        HLA_length_aatype_position_softmax_pd = np.array(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, length + 1))
        
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd,
                                                               index = aa_indexs, columns = range(1, length + 1))
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd
    
def HLA_Length_aatype_position_num(data, attn_data, hla = 'HLA-A*11:01', label = None, length = 9):
 
    if label == None:
        length_index = np.array(data[data.length == length][data.HLA == hla].index)
    else:
        length_index = np.array(data[data.length == length][data.HLA == hla][data.label == label].index)

    aatype_position_num = dict()
    for idx in length_index:
        temp_peptide = data.iloc[idx].peptide
        for i, aa in enumerate(temp_peptide):
            aatype_position_num.setdefault(aa, {})
            aatype_position_num[aa].setdefault(i, 0)
            aatype_position_num[aa][i] += 1         
    
    ####################
    aatype_position_num_pd = np.zeros((20, length))
    
    aai, aa_indexs = 0, []
    for aa, aa_posi in aatype_position_num.items():
        aa_indexs.append(aa)
        for posi, v in aa_posi.items():
            aatype_position_num_pd[aai, posi] = v
        aai += 1
        
    ####################
    if len(aa_indexs) != 20: 
        aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
        abscent_aa = list(set(aatype_sorts).difference(set(aa_indexs)))
        aa_indexs += abscent_aa
        
    #####################    
    aatype_position_num_unsoftmax_pd = pd.DataFrame(aatype_position_num_pd,
                                                           index = aa_indexs, columns = range(1, length + 1))
    
    aatype_position_num_unsoftmax_pd.loc['sum'] = aatype_position_num_unsoftmax_pd.sum(axis = 0)
    aatype_position_num_unsoftmax_pd['sum'] = aatype_position_num_unsoftmax_pd.sum(axis = 1)
    
    return aatype_position_num_unsoftmax_pd.astype('int')
    
def draw_hla_length_aatype_position(data, attn_data, hla = 'HLA-B*27:05', label = None, length = 9, 
                                    show = True, softmax = True, unsoftmax = True):
    
    HLA_length_aatype_position = attn_HLA_length_aatype_position_num(data, attn_data, hla, label, length, show_num = False)
    
    if softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd = attn_HLA_length_aatype_position_pd(
                                                                                     HLA_length_aatype_position, 
                                                                                     length, 
                                                                                     softmax,
                                                                                     unsoftmax)
        HLA_length_aatype_position_softmax_pd = sort_aatype(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_unsoftmax_pd = sort_aatype(HLA_length_aatype_position_unsoftmax_pd)
        
        if show:
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 8))
            sns.heatmap(HLA_length_aatype_position_softmax_pd,
                        ax = axes[0], cmap = 'YlGn', square = True)

            sns.heatmap(HLA_length_aatype_position_unsoftmax_pd,
                        ax = axes[1], cmap = 'YlGn', square = True)

            axes[0].set_title(hla + ' Softmax Normalization')
            axes[1].set_title(hla + ' UnNormalization')
            plt.show()

        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd
    
    else:
        HLA_length_aatype_position_pd = attn_HLA_length_aatype_position_pd(HLA_length_aatype_position, 
                                                                           length, 
                                                                           softmax,
                                                                           unsoftmax)
        HLA_length_aatype_position_pd = sort_aatype(HLA_length_aatype_position_pd)
        return HLA_length_aatype_position_pd
    

# peptide aatype-peptide position
def attn_pepaatype_pepposition(data, attn, label = None, length = None):
    aatype_position = dict()
    
    if label == None and length == None:
        length_index = np.array(data.index)
    elif label!= None and length == None:
        length_index = np.array(data[data.label == label].index)
    elif label == None and length != None:
        length_index = np.array(data[data.length == length].index)
    elif label != None and length != None:
        length_index = np.array(data[data.label == label][data.length == length].index)
    
    length_data_num = len(length_index)
    print(length_data_num)

    for head in range(9):
        for idx in length_index:
            temp_peptide = data.iloc[idx].peptide
            temp_length_head = deepcopy(nn.Softmax(dim=-1)(attn[idx][head][:, :l].float())) # Shape = (34, 14), 行是HLA，列是peptide，由行查列
            temp_length_head = nn.Softmax(dim=-1)(temp_length_head.sum(axis = 0)) # 把这一列的数据相加，shape = （14，）

            for i, aa in enumerate(temp_peptide): 
                aatype_position.setdefault(aa, {})
                aatype_position[aa].setdefault(i, 0)
                aatype_position[aa][i] += temp_length_head[i] 
    return aatype_position

def attn_aatype_position_pds(attn_aatype_position, length = 14, softmax = True, unsoftmax = True):
    
    if softmax:
        attn_aatype_position_Softmax_pd = np.zeros((20, length))
        aai, aa_indexs = 0, []
        for aa, aa_posi in attn_aatype_position.items():
            aa_indexs.append(aa)
            for posi, v in aa_posi.items():
                attn_aatype_position_Softmax_pd[aai, posi] = v
            aai += 1

        attn_aatype_position_Softmax_pd = deepcopy(nn.Softmax(dim = -1)(torch.Tensor(attn_aatype_position_Softmax_pd)))
        attn_aatype_position_Softmax_pd = np.array(attn_aatype_position_Softmax_pd)

        attn_aatype_position_Softmax_pd = pd.DataFrame(attn_aatype_position_Softmax_pd, index = aa_indexs, columns = range(1, length + 1))

    if unsoftmax:
        attn_aatype_position_pd = np.zeros((20, length))
        aai, aa_indexs = 0, []
        for aa, aa_posi in attn_aatype_position.items():
            aa_indexs.append(aa)
            for posi, v in aa_posi.items():
                attn_aatype_position_pd[aai, posi] = v
            aai += 1

        attn_aatype_position_pd = pd.DataFrame(attn_aatype_position_pd, index = aa_indexs, columns = range(1, length + 1))
    
    if softmax and unsoftmax:
        return sort_aatype(attn_aatype_position_Softmax_pd), sort_aatype(attn_aatype_position_pd)
    if softmax and not unsoftmax:
        return sort_aatype(attn_aatype_position_Softmax_pd)
    if not softmax and unsoftmax:
        return sort_aatype(attn_aatype_position_pd)

def Length_Label_pepaatype_pepposition(data, attn_data, length):
    softmax = False
    unsoftmax = True
    new_length = [length, 14][length == None]
        
    pepaatype_pepposition = attn_pepaatype_pepposition(data, attn_data, label = None, length = length)
    pepaatype_pepposition_pd = attn_aatype_position_pds(pepaatype_pepposition, new_length, softmax, unsoftmax)

    positive_pepaatype_pepposition = attn_pepaatype_pepposition(data, attn_data, label = 1, length = length)
    positive_pepaatype_pepposition_pd = attn_aatype_position_pds(positive_pepaatype_pepposition, new_length, softmax, unsoftmax)

    negative_pepaatype_pepposition = attn_pepaatype_pepposition(data, attn_data, label = 0, length = length)
    negative_pepaatype_pepposition_pd = attn_aatype_position_pds(negative_pepaatype_pepposition, new_length, softmax, unsoftmax)
    
    return pepaatype_pepposition_pd, positive_pepaatype_pepposition_pd, negative_pepaatype_pepposition_pd

def Length_pepaatype_pepposition_num(data, attn_data, label = None, length = None):
 
    if label == None and length == None:
        length_index = np.array(data.index)
    elif label!= None and length == None:
        length_index = np.array(data[data.label == label].index)
    elif label == None and length != None:
        length_index = np.array(data[data.length == length].index)
    elif label != None and length != None:
        length_index = np.array(data[data.label == label][data.length == length].index)

    aatype_position_num = dict()
    for idx in length_index:
        temp_peptide = data.iloc[idx].peptide
        for i, aa in enumerate(temp_peptide):
            aatype_position_num.setdefault(aa, {})
            aatype_position_num[aa].setdefault(i, 0)
            aatype_position_num[aa][i] += 1         
    
    ####################
    new_length = [length, 14][length == None]
    aatype_position_num_pd = np.zeros((20, new_length))
    
    aai, aa_indexs = 0, []
    for aa, aa_posi in aatype_position_num.items():
        aa_indexs.append(aa)
        for posi, v in aa_posi.items():
            aatype_position_num_pd[aai, posi] = v
        aai += 1
        
    ####################
    if len(aa_indexs) != 20: 
        aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
        abscent_aa = list(set(aatype_sorts).difference(set(aa_indexs)))
        aa_indexs += abscent_aa
        
    #####################    
    aatype_position_num_unsoftmax_pd = pd.DataFrame(aatype_position_num_pd,
                                                           index = aa_indexs, columns = range(1, new_length + 1))
    
    aatype_position_num_unsoftmax_pd.loc['sum'] = aatype_position_num_unsoftmax_pd.sum(axis = 0)
    aatype_position_num_unsoftmax_pd['sum'] = aatype_position_num_unsoftmax_pd.sum(axis = 1)
    
    return aatype_position_num_unsoftmax_pd.astype('int')

def draw_Length_Label_pepaatype_pepposition(pepaatype_pepposition_pd, 
                                            positive_pepaatype_pepposition_pd, 
                                            negative_pepaatype_pepposition_pd, 
                                            length = None, savepath = True, softmax = False): 
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5), dpi = 600)
    fig.patch.set_facecolor('white')
    cmap = 'YlOrRd'

    sns.heatmap(pepaatype_pepposition_pd, ax = axes[0], cmap = cmap, square = True, xticklabels = True, yticklabels = True)
    axes[0].set_title('All samples')

    sns.heatmap(positive_pepaatype_pepposition_pd, ax = axes[1], cmap = cmap, square = True, xticklabels = True, yticklabels = True)
    axes[1].set_title('Positive samples')

    sns.heatmap(negative_pepaatype_pepposition_pd, ax = axes[2], cmap = cmap, square = True, xticklabels = True, yticklabels = True)
    axes[2].set_title('Negative samples')

    axes[0].set_ylabel('Amino acid type of peptides')
    axes[1].set_xlabel('Peptide position')

    fig.suptitle('Length = {} | Amino acid types of peptides - Peptide positions'.format(length))
    
    softmax = ['Softmax', 'Unsoftmax'][softmax == False]
    if savepath:
        plt.savefig('./figures/pHLAIformer/Attention/Length{}_{}_Label_AAtypes_PepPositions.tif'.format(length, softmax),
                    dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())
    plt.show()

# 对于每个HLA和每个length，生成AA type - peptide positon
def draw2rows_HLA_length_aatype_position(hla, data, attn_data, length = 9):
    HLA_length_aatype_position_Softmax_pd, HLA_length_aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, show = False, label = None, length = length)
    HLA_length_Positive_aatype_position_Softmax_pd, HLA_length_Positive_aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, show = False, label = 1, length = length)
    HLA_length_Negative_aatype_position_Softmax_pd, HLA_length_Negative_aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, show = False, label = 0, length = length)
    
    ##############################
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))
    fig.patch.set_facecolor('white')
    cmap = 'cividis'
    
    sns.heatmap(HLA_length_aatype_position_Softmax_pd, ax = axes[0], cmap = cmap, square = True, cbar = False)
    axes[0].set_title('All samples')

    sns.heatmap(HLA_length_Positive_aatype_position_Softmax_pd, ax = axes[1], cmap = cmap, square = True, cbar = False)
    axes[1].set_title('Positive samples')

    sns.heatmap(HLA_length_Negative_aatype_position_Softmax_pd, ax = axes[2], cmap = cmap, square = True, cbar = False)
    axes[2].set_title('Negative samples')

    norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), 
                        orientation='vertical', ax = axes, shrink = 1, anchor = (-0.3, 0.5))
    cbar.outline.set_visible(False)
    
    fig.suptitle('Peptide length = 9 | {} | Softmax Normalization | Amino acid types of peptides - Peptide positions'.format(hla))
#     plt.text(x = -30, y = -2, s = '(a) Softmax Normalization')
    axes[0].set_ylabel('Amino acid type of peptides')
    axes[1].set_xlabel('Peptide position')
    plt.savefig('./figures/pHLAIformer/Attention/Length{}_{}_Amino acid types of peptides - Peptide positions.tif'.format(length, hla), 
                dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())
    plt.show()
    #######################
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10.5, 5))
    fig.patch.set_facecolor('white')

    sns.heatmap(HLA_length_aatype_position_pd, ax = axes[0], cmap = cmap, square = True)
    axes[0].set_title('All samples')

    sns.heatmap(HLA_length_Positive_aatype_position_pd, ax = axes[1], cmap = cmap, square = True)
    axes[1].set_title('Positive samples')

    sns.heatmap(HLA_length_Negative_aatype_position_pd, ax = axes[2], cmap = cmap, square = True)
    axes[2].set_title('Negative samples')

#     plt.text(x = -30, y = -2, s = '(b) UnNormalization')
    fig.suptitle('Peptide length = 9 | {} | UnNormalization | Amino acid types of peptides - Peptide positions'.format(hla),)
    axes[0].set_ylabel('Amino acid type of peptides')
    axes[1].set_xlabel('Peptide position')

    plt.savefig('./figures/pHLAIformer/Attention/Length{}_{}_PepAAtypes_PepPositions_UnSoftmax.tif'.format(length, hla), 
                dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())
    plt.show()
    
def draw1rows_HLA_length_pepaatype_pepposition(hla, data, attn_data):
    HLA_length_aatype_position_Softmax_pd, HLA_length_aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, show = False)
    HLA_length_Positive_aatype_position_Softmax_pd, HLA_length_Positive_aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, show = False, label = 1)
    HLA_length_Negative_aatype_position_Softmax_pd, HLA_length_Negative_aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, show = False, label = 0)
    
    ##############################
    fig, axes = plt.subplots(nrows = 1, ncols = 6, figsize = (24, 6.5), dpi = 600)
    fig.patch.set_facecolor('white')
    cmap = 'cividis'
    
    sns.heatmap(HLA_length_aatype_position_Softmax_pd, ax = axes[0], cmap = cmap, square = True, cbar = False)
    axes[0].set_title('All samples')

    sns.heatmap(HLA_length_Positive_aatype_position_Softmax_pd, ax = axes[1], cmap = cmap, square = True, cbar = False)
    axes[1].set_title('Positive samples')

    sns.heatmap(HLA_length_Negative_aatype_position_Softmax_pd, ax = axes[2], cmap = cmap, square = True, cbar = False)
    axes[2].set_title('Negative samples')

    norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), 
                        orientation='vertical', ax = axes[:3], shrink = 1, anchor = (-0.3, 0.5))
    cbar.outline.set_visible(False)

    plt.text(x = -55.5, y = -1.5, s = 'Softmax Normalization')
    #######################
    sns.heatmap(HLA_length_aatype_position_pd, ax = axes[3], cmap = cmap, square = True)
    axes[3].set_title('All samples')

    sns.heatmap(HLA_length_Positive_aatype_position_pd, ax = axes[4], cmap = cmap, square = True)
    axes[4].set_title('Positive samples')

    sns.heatmap(HLA_length_Negative_aatype_position_pd, ax = axes[5], cmap = cmap, square = True)
    axes[5].set_title('Negative samples')
    plt.text(x = -11.5, y = -1.5, s = 'UnNormalization', fontsize = 12)
    #######################
    axes[0].set_ylabel('Amino acid type of peptides')
    axes[3].set_ylabel('Amino acid type of peptides')
    axes[1].set_xlabel('Peptide position')
    axes[4].set_xlabel('Peptide position')
    
    fig.suptitle('Peptide length = 9 | {} | Amino acid types of peptides - Peptide positions'.format(hla))
    plt.savefig('./figures/pHLAIformer/Attention/Length9_{}_PepAAtypes_PepPositions.tif'.format(hla), 
                dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())
    plt.show()
    
def draw_Unsoftmax_HLA_Length_aatype_position(data, attn_data, hla, length = 9, show = False):
    
    aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, label = None, length = length, show = False, softmax = False, unsoftmax = True)
    Positive_aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, label = 1, length = length, show = False, softmax = False, unsoftmax = True)
    Negative_aatype_position_pd = draw_hla_length_aatype_position(data, attn_data, hla, label = 0, length = length, show = False, softmax = False, unsoftmax = True)

    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10.5, 5))
    fig.patch.set_facecolor('white')
    
    cmap = 'YlOrRd'
    sns.heatmap(aatype_position_pd, ax = axes[0], cmap = cmap, square = True)
    axes[0].set_title('All samples')

    sns.heatmap(Positive_aatype_position_pd, ax = axes[1], cmap = cmap, square = True)
    axes[1].set_title('Positive samples')

    sns.heatmap(Negative_aatype_position_pd, ax = axes[2], cmap = cmap, square = True)
    axes[2].set_title('Negative samples')

    fig.suptitle('Peptide length = {} | {} | Amino acid types of peptides - Peptide positions'.format(peptide, hla))
    axes[0].set_ylabel('Amino acid type of peptides')
    axes[1].set_xlabel('Peptide position')

    plt.savefig('./figures/pHLAIformer/Attention/Length{}_{}_PepAAtypes_PepPositions_UnSoftmax.tif'.format(length, hla), 
                dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())
    if show: plt.show()
    return aatype_position_pd, Positive_aatype_position_pd, Negative_aatype_position_pd

def draw_hla_length_aatype_position(hla = 'HLA-B*27:05', length = 9, savepath = False):
    
    hla_length_aatype_peppsition = np.load('./results/Attention/pHLAIformer_{}_Length{}_Label_Unsoftmax_pepaatype_pepposition_pd.npy'.format(hla, length), allow_pickle = True).item()
    
    all_data = hla_length_aatype_peppsition['all']
    positive_data = hla_length_aatype_peppsition['positive']
    negative_data = hla_length_aatype_peppsition['negative']

    cmap = 'YlGn'
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 8))
    sns.heatmap(all_data, ax = axes[0], cmap = cmap, square = True, xticklabels = True, yticklabels = True, cbar = True)
    sns.heatmap(positive_data, ax = axes[1], cmap = cmap, square = True, xticklabels = True, yticklabels = True, cbar = True)
    sns.heatmap(negative_data, ax = axes[2], cmap = cmap, square = True, xticklabels = True, yticklabels = True, cbar = True)

    axes[0].set_title('All samples')
    axes[1].set_title('Positive samples')
    axes[2].set_title('Negative samples')
    
    axes[0].set_ylabel('Amino acid type of peptide')
    axes[1].set_xlabel('peptide position')
    fig.suptitle('{} | Length = {} | Amino acid types of peptide - peptide positions'.format(hla, length))
    
    if savepath:
        plt.savefig(show_savepath + '{}_{}_peptideAAtype_pepposition.tif'.format(hla, peptide),
                    dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())
    plt.show()
    

# peptide-HLA sequence

def draw_pHLA_attns(data, attn_data, hla = 'HLA-A*02:01', peptide = 'SISELVAYL', savepath = False):
    
    idx = data[data.HLA == hla][data.peptide == peptide].index[0]
    hla_sequence = list(data.iloc[idx].HLA_sequence)
    label = data.iloc[idx].label
    length = len(peptide)
    for head in range(9):
        if head == 0:
            temp = deepcopy(attn_data[idx][head][:, :length])
            temp_softmax = deepcopy(nn.Softmax(dim=-1)(attn_data[idx][head][:, :length].float())) # Shape = (34, 14), 行是HLA，列是peptide，由行查列
        else:
            temp += deepcopy(attn_data[idx][head][:, :length])
            temp_softmax += deepcopy(nn.Softmax(dim=-1)(attn_data[idx][head][:, :length].float()))

    temp_softmax = deepcopy(nn.Softmax(dim = -1)(temp_softmax))

    temp_pd = pd.DataFrame(np.array(temp), index = hla_sequence, columns = list(peptide)).T
    temp_softmax_pd = pd.DataFrame(np.array(temp_softmax), index = list(hla_sequence), columns = list(peptide)).T
    ###############################
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 2.5))
    fig.patch.set_facecolor('white')
    cmap = 'cividis'
    sns.heatmap(temp_pd, ax = axes[0], cmap = cmap, square = True, cbar = False)
    sns.heatmap(temp_softmax_pd, ax = axes[1], cmap = cmap, square = True, cbar = False)
    axes[0].set_title('UnNormalization')
    axes[1].set_title('Softmax Normalization')
    axes[0].set_ylabel('Peptide')
    axes[0].set_xlabel('HLA')
    axes[1].set_xlabel('HLA')
    fig.suptitle('{} | {} | {}'.format(hla, peptide, ['Negative', 'Positive'][label == 1]))
    
    if savepath:
        plt.savefig('./figures/pHLAIformer/Attention/{}_{}_{}.tif'.format(hla, peptide, ['Negative', 'Positive'][label == 1]),
                    dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())

    plt.show()
    
    
def pHLA_attns_draw_save(data, attn_data, hla = False, peptide = False, attn_savepath = False, fig_savepath = False):
    
    idx = data[data.HLA == hla][data.peptide == peptide].index[0]
    hla_sequence = list(data.iloc[idx].HLA_sequence)
    y_pred = data.iloc[idx].y_pred
    y_prob = data.iloc[idx].y_prob.round(4)
    length = len(peptide)
    
    for head in range(9):
        if head == 0:
            temp = deepcopy(attn_data[idx][head][:, :length])
        else:
            temp += deepcopy(attn_data[idx][head][:, :length])

    temp_pd = pd.DataFrame(np.array(temp), index = hla_sequence, columns = list(peptide))
    temp_pd.loc['sum'] = temp_pd.sum(axis = 0)
    temp_pd.loc['posi'] = range(1, length + 1)
    temp_pd.loc['contrib'] = temp_pd.loc['sum'] / temp_pd.loc['sum'].sum()
        
    if attn_savepath: 
        temp_pd.to_csv(attn_savepath + '{}_{}_attention.csv'.format(hla, peptide))
        
    ###############################
    
    if fig_savepath:
        fig = plt.figure(figsize = (10, 3))
        fig.patch.set_facecolor('white')

        cmap = 'cividis'
        sns.heatmap(temp_pd.iloc[:-3].T, cmap = cmap, square = True,  
                    xticklabels = True, yticklabels = True, cbar = True)
        plt.ylabel('Peptide')
        plt.xlabel('HLA')
        plt.title('{} | {} | {} | Probability = {}'.format(hla, peptide, ['Negative', 'Positive'][y_pred == 1], y_prob))

        plt.savefig(fig_savepath + '{}_{}_{}_Prob{}.tif'.format(hla, peptide, ['Negative', 'Positive'][y_pred == 1], y_prob),
                dpi = 600, pil_kwargs = {'compression': 'tiff_lzw'}, bbox_inches = 'tight', facecolor=fig.get_facecolor())

    return temp_pd, hla, peptide