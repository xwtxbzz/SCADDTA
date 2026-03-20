from rdkit import Chem
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
from skfp.fingerprints import *

def get_fingerprint(smiles):
    mol_from_smiles = MolFromSmilesTransformer()
    mols = mol_from_smiles.transform([smiles])
    fp = ERGFingerprint()
    ergfp = fp.transform(mols)[0]
    fp = PubChemFingerprint()
    pubfp = fp.transform(mols)[0]
    fp = MACCSFingerprint()
    maccsfp = fp.transform(mols)[0]
    if len(smiles) >= 500:
        return (np.zeros(1024), ergfp, pubfp, maccsfp)
    try:
        conf_gen = ConformerGenerator()
        mols = conf_gen.transform(mols)
        fp = E3FPFingerprint()
        e3fp = fp.transform(mols)[0]
    except:
        e3fp = np.zeros(1024)
        
    return (e3fp, ergfp, pubfp, maccsfp)

import pandas as  pd
import numpy as np
import os
dataset = 'ttd/IC50'
# smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
drug = pd.read_csv("./datasets/{}/smiles.csv".format(dataset)).to_numpy()
smiles = ['DB01097','CC1=C(C=NO1)C(=O)NC1=CC=C(C=C1)C(F)(F)F']
e3fp,ergfp,pubfp,maccsfp = get_fingerprint(smiles[1])
np.savez('./datasets/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', smiles[0]), e3fp = e3fp,ergfp = ergfp,pubfp = pubfp,maccsfp = maccsfp)
for idx,s in enumerate(drug):
    print(s[0])
    try:
        fig = np.load('./datasets/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0]), allow_pickle=True)
    except:
        e3fp,ergfp,pubfp,maccsfp = get_fingerprint(s[1])
        np.savez('./datasets/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0]), e3fp = e3fp,ergfp = ergfp,pubfp = pubfp,maccsfp = maccsfp)
    if len(s[1]) > 500:
        try:
            finger = np.load('./datasets/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0]), allow_pickle=True)
            np.savez('./datasets/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0]), e3fp = finger['e3fp'],ergfp = finger['ergfp'],pubfp = finger['pubfp'],maccsfp = finger['maccsfp'])
        except:
            e3fp,ergfp,pubfp,maccsfp = get_fingerprint(s[1])
            np.savez('./datasets/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0]), e3fp = e3fp,ergfp = ergfp,pubfp = pubfp,maccsfp = maccsfp)
            continue
    if os.path.exists('./datasets/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0])):
        continue
    e3fp,ergfp,pubfp,maccsfp = get_fingerprint(s[1])
    np.savez('./datasets/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0]), e3fp = e3fp,ergfp = ergfp,pubfp = pubfp,maccsfp = maccsfp)