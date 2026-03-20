import torch
from torch import nn
from tqdm.notebook import tqdm
from metric import   get_rm2, get_ci, get_mse, get_rmse, get_pearson, get_spearman
import os
import numpy as np
import pandas as pd
import time
from model import CoreNet
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from data import DTAData, cold_start_split, drug_cold_start_split, target_cold_start_split
def calculate_metrics(Y, P, dataset='davis',type = 'test'):
    cindex2 = get_ci(Y, P) 
    rm2 = get_rm2(Y, P)  
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    result_file_name = './results/result_' + '_' + dataset + '.txt'
    result_str = ''
    result_str += '\n'+type+' '+dataset + '\r\n'
    result_str += 'rmse:' + str(rmse) + ' ' + ' mse:' + str(mse) + ' ' + ' pearson:' + str(
        pearson) + ' ' + 'spearman:' + str(spearman) + ' ' + 'ci:' + str(cindex2) + ' ' + 'rm2:' + str(rm2)
    print(result_str)
    open(result_file_name, 'a').writelines(result_str)
dataset = 'davis'
start = "cold_target" # "cold_target" "cold"  "cold_drug"
data = pd.read_csv("./datasets/{}/affinity.csv".format(dataset))
if start =='cold_drug':
    print("cold_drug")
    train_data,test_data = drug_cold_start_split(data, test_size=0.16667, random_state=42)
elif start == "cold_target":
    print("cold_target")
    train_data,test_data = target_cold_start_split(data, test_size=0.16667, random_state=42)
elif start == "cold":
    print("cold")
    train_data,test_data = cold_start_split(data, test_size=0.16667, random_state=42)
else:
    print("full")
    data = data.to_numpy()[:]
    train_data, test_data = train_test_split(data, test_size=0.16667, random_state=42, shuffle=False) # 42
print(train_data.shape,test_data.shape)
train = DTAData(train_data,dataset_name=dataset)
train_data = DataLoader(train,batch_size = 128,shuffle = True)
test = DTAData(test_data,dataset_name=dataset)
test_data = DataLoader(test,batch_size =  128)
if len(dataset.split('/')) > 1:
        dataset = dataset.split('/')[-1]
model = CoreNet(
            n_output=1,
            output_dim=256,
            dropout=0.1,
            num_features_mol=78,
            num_features_pro= 34 if dataset=='IC50' else 33
        )
print(dataset)
model_name = 'HCoreNet'
if start in ['cold_drug', 'cold_target', 'cold']:
    model_name += '_' + start
if os.path.exists('./model/{}_{}.pth'.format(model_name,dataset)):
    model.load_state_dict(torch.load('./model/{}_{}.pth'.
                                    format(model_name,dataset)))
optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-6,
)  
scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.999
)
        
criterion = nn.MSELoss() 
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_step = 0
torch.cuda.empty_cache()
avg_loss = 0
for epoch in range(500):
    train_pbar = tqdm(train_data, desc=f'Training Epoch {epoch}', leave=False,ascii=True)
    avg_loss = 0
    torch.cuda.empty_cache()
    model.train()  
    for i, (s_smiles, s_sequence, value, e3fp, ergfp, pubfp, maccsfp, data_mol, data_pro, smi_embed, seq_embed) in enumerate(train_pbar):
        optimizer.zero_grad()
        ti = time.time()
        output,concept = model(
                    s_smiles.cuda(),
                    s_sequence.cuda(),
                    e3fp.cuda(),
                    ergfp.cuda(),
                    pubfp.cuda(),
                    maccsfp.cuda(),
                    data_mol.cuda(),
                    data_pro.cuda(),
                    smi_embed.cuda(),
                    seq_embed.cuda(),
                    )
        loss = criterion(output, torch.FloatTensor(value).cuda())
        avg_loss += loss.item()

        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'time': f'{time.time() - ti:.2f}s',
            'batch': i
        })

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    scheduler.step()    
    print('Average Loss:', avg_loss / len(train_data))
    torch.save(model.state_dict(), './model/{}_{}.pth'.format(model_name,dataset))

    y = np.array([])
    p = np.array([])
    model.eval()
    with torch.no_grad():
        test_pbar = tqdm(test_data, desc='Testing', leave=False,ascii=True)
        for s_smiles, s_sequence, value, e3fp, ergfp, pubfp, maccsfp, data_mol, data_pro, smi_embed, seq_embed in test_pbar:
            torch.cuda.empty_cache()
        
            output,concept = model(
                        s_smiles.cuda(),
                        s_sequence.cuda(),
                        e3fp.cuda(),
                        ergfp.cuda(),
                        pubfp.cuda(),
                        maccsfp.cuda(),
                        data_mol.cuda(),
                        data_pro.cuda(),
                        smi_embed.cuda(),
                        seq_embed.cuda()
                    )
            y = np.concatenate((y, value.flatten()))
            p = np.concatenate((p, output.flatten().cpu().detach().numpy()))

            test_pbar.set_postfix({
                'samples': len(y)
            })

    calculate_metrics(np.array(y), np.array(p), dataset=dataset+"_"+model_name,type='test')