import pandas as pd
import zipfile
import os
import logging
from logging import handlers

def HLA_Label_Length_Position_AA_peptides(data, hla, label, length, posi, aa):
    posi -= 1
    peptides = [item for item in data[data.HLA == hla][data.label == label][data.length == length].peptide 
                     if item[posi] == aa]
    return peptides

def HLA_Length_Position_AA_peptides(data, hla, length, pos_posi, neg_posi, pos_aa, neg_aa):
    peptides_Positive = HLA_Label_Length_Position_AA_peptides(data, hla, 1, length, pos_posi, pos_aa)
    peptides_Negative = HLA_Label_Length_Position_AA_peptides(data, hla, 0, length, neg_posi, neg_aa)
    print('{} | Length = {} | Positive Peptide Position = {} | Positive AA = {} | Negative Peptide Position = {} | Negative AA = {}'.format(hla, length, pos_posi, pos_aa, neg_posi, neg_aa))
    print('# Positive = {} | # Negative = {}'.format(len(peptides_Positive), len(peptides_Negative)))
    return peptides_Positive, peptides_Negative

def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')    
    pre_len = len(os.path.dirname(source_dir))
    for parent, _, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    zipf.close()

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',fmt='%(asctime)s : %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.FileHandler(filename, mode = 'w')#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        self.logger.addHandler(sh) #把对象加到logger里
        
def cut_peptide_to_specific_length(peptide, cut_length):
    length = len(peptide)
    if length > cut_length:
        cut_peptides = [peptide[i : i + cut_length] for i in range(length - cut_length + 1)]
        return cut_peptides
    else:
        return peptide