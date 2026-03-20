import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data,InMemoryDataset, Batch

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
def cold_start_split(data, test_size=0.2, random_state=42):
    unique_drugs = sorted(list(set(data['DRUGID'])))
    unique_targets = sorted(list(set(data['TARGETID'])))
    train_drugs, test_drugs = train_test_split(unique_drugs, test_size=test_size, random_state=random_state)
    train_targets, test_targets = train_test_split(unique_targets, test_size=test_size, random_state=random_state)
    train_mask = data['DRUGID'].isin(train_drugs) & data['TARGETID'].isin(train_targets)
    train_data = data[train_mask]
    test_mask = data['DRUGID'].isin(test_drugs) & data['TARGETID'].isin(test_targets)
    test_data = data[test_mask]

    return train_data, test_data
def drug_cold_start_split(data, test_size=0.2, random_state=42):
    unique_drugs = sorted(list(set(data['DRUGID'])))
    train_drugs, test_drugs = train_test_split(unique_drugs, test_size=test_size, random_state=random_state)
    train_mask = data['DRUGID'].isin(train_drugs)
    train_data = data[train_mask]
    test_mask = data['DRUGID'].isin(test_drugs)
    test_data = data[test_mask]

    return train_data, test_data
def target_cold_start_split(data, test_size=0.2, random_state=42):
    unique_targets = sorted(list(set(data['TARGETID'])))
    train_targets, test_targets = train_test_split(unique_targets, test_size=test_size, random_state=random_state)
    train_mask = data['TARGETID'].isin(train_targets)
    train_data = data[train_mask]
    test_mask = data['TARGETID'].isin(test_targets)
    test_data = data[test_mask]

    return train_data, test_data
def label_chars(chars, char_set,len):
    X = np.zeros(len, dtype=np.int32)
    for i, ch in enumerate(chars):
        X[i] = char_set[ch]
    return X
CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
            "U": 19, "T": 20, "W": 21, 
            "V": 22, "Y": 23, "X": 24, 
            "Z": 25 }

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
class DTAData(InMemoryDataset):
    def __init__(self,aff,dataset_name = 'davis'):
        self.seq = pd.read_csv("./datasets/{}/sequence.csv".format(dataset_name))
        self.smi = pd.read_csv("./datasets/{}/smiles.csv".format(dataset_name))
        self.aff = np.array(aff)
        self.dataset = dataset_name
        self.len = len(self.aff)
    def __getitem__(self, index):
        drug_id = self.aff[index][0]
        target_id = self.aff[index][1]
        if self.dataset == 'davis' or self.dataset == 'metz':  
            smiles = self.smi[self.smi['DRUGID']==int(drug_id)]['SMILES'].values[0]
        else:
            smiles = self.smi[self.smi['DRUGID']==drug_id]['SMILES'].values[0]
        if len(smiles)>1024: smiles = smiles[:1024]
        seq =  self.seq[self.seq['TARGETID']==self.aff[index][1]][:1024]
        sequence =seq['SEQUENCE'].values[0][:1024]
        value = torch.Tensor([float(self.aff[index][2])])
        
        s_smiles = torch.LongTensor(label_chars(smiles, CHARISOSMISET,1024))
        s_sequence = torch.LongTensor(label_chars(sequence, CHARPROTSET,1024))
        
        d_features = np.load('./datasets/{}/{}/{}.npz'.format(self.dataset,'drug_graph', drug_id), allow_pickle=True)
        _, smiles_features, smiles_edge_index = d_features['smiles_features'].shape[0], d_features['smiles_features'], d_features['smiles_edge_index']
        t_features = np.load('./datasets/{}/{}/{}.npz'.format(self.dataset,'protein_graph', target_id), allow_pickle=True)
        _, target_features, target_edge_index = t_features['target_features'].shape[0], t_features['target_features'], t_features['target_edge_index']
        smile_graph = Data(x=torch.Tensor(smiles_features),edge_index=torch.LongTensor(smiles_edge_index).transpose(1,0),y=value)
        target_graph = Data(x=torch.Tensor(target_features), edge_index=torch.LongTensor(target_edge_index).transpose(1,0),y=value)
        chars = t_features['target_m'][0]
        X = np.pad(chars, ((0,1024-len(chars)), (0, 0)), mode='constant', constant_values=0)
        seq_embed = torch.Tensor(X)
        chars = d_features['smiles_m'][0]
        X = np.pad(chars, ((0,1024-len(chars)), (0, 0)), mode='constant', constant_values=0)
        smi_embed = torch.Tensor(X)
        
        
        finger = np.load('./datasets/{}/{}/{}.npz'.format(self.dataset,'drug_fingerprints', drug_id), allow_pickle=True)
        e3fp, ergfp, pubfp, maccs =  finger['e3fp'], finger['ergfp'], finger['pubfp'], finger['maccsfp']
        e3fp, ergfp, pubfp, maccs = torch.Tensor(e3fp), torch.Tensor(ergfp), torch.Tensor(pubfp), torch.Tensor(maccs)
        
        
        return  s_smiles, s_sequence,value,e3fp,ergfp,pubfp,maccs,smile_graph,target_graph,smi_embed,seq_embed
    
    def __len__(self):
        return self.len
    
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    batchE = Batch.from_data_list([data[4] for data in data_list]) 
    batchF = Batch.from_data_list([data[5] for data in data_list]) 
    batchG = Batch.from_data_list([data[6] for data in data_list])
    batchH = Batch.from_data_list([data[7] for data in data_list])
    batchI = Batch.from_data_list([data[8] for data in data_list])
    batchJ = Batch.from_data_list([data[9] for data in data_list])
    batchK = Batch.from_data_list([data[10] for data in data_list])
    return (batchA, batchB,batchC,batchD,batchE,batchF,batchG,batchH,batchI,batchJ,batchK)