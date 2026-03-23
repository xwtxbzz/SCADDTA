import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SuperGATConv, global_mean_pool as gep


class DynamicCrossModalAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_modalities=6, num_heads=4, dropout=0.1):
        super(DynamicCrossModalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.content_weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities * num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, modalities_features):
        batch_size = modalities_features.size(0)
        qkv = self.qkv(modalities_features).reshape(batch_size, self.num_modalities, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_modalities, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        content_weights = self.content_weight_predictor(
            modalities_features.reshape(batch_size, -1)
        ).reshape(batch_size, self.num_modalities, self.num_modalities)
        attn_weights = attn_weights + content_weights.unsqueeze(1)  # broadcast to all heads
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)  # (batch_size, num_heads, num_modalities, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, self.num_modalities, self.hidden_dim)
        
        out = self.proj(out)
        out = out + modalities_features
        out = self.layer_norm(out)
        
        return out, attn_weights


class SphereBatchNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super(SphereBatchNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, x):        
        mean = torch.mean(x, dim=0, keepdim=True)
        x_centered = x - mean
        std = torch.sqrt(torch.var(x, dim=0, keepdim=True) + self.eps)
        x_normalized = x_centered / std
        return x_normalized * self.gamma + self.beta


class SphericalConstrainedProjectionFusion(nn.Module):
    def __init__(self, input_dims,hidden_dim=256, output_dim=256, dropout=0.1):
        super(SphericalConstrainedProjectionFusion, self).__init__()
        
        self.input_dims = input_dims
        self.modality_names = list(input_dims.keys())
        self.hidden_dim = hidden_dim
        
        self.euc_to_hyp = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.euc_to_hyp[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
        
        total_hidden = hidden_dim * len(input_dims)
        self.hyp_fusion = nn.Sequential(
            nn.Linear(total_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hyp_bn = SphereBatchNorm(hidden_dim)
        self.hyp_to_euc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, modality_features):
        hyperbolic_features = []
        for modality in self.modality_names:
            feat = self.euc_to_hyp[modality](modality_features[modality])
            feat = feat / (1 + torch.norm(feat, dim=-1, keepdim=True))
            hyperbolic_features.append(feat)
        
        concat_hyp_features = torch.cat(hyperbolic_features, dim=-1)
        fused_hyp = self.hyp_fusion(concat_hyp_features)
        fused_hyp = self.hyp_bn(fused_hyp)
        output = self.hyp_to_euc(fused_hyp)
        
        return output


class AdaptiveMultiModalFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=512, output_dim=256, num_heads=4, dropout=0.1):
        super(AdaptiveMultiModalFusion, self).__init__()
        
        self.input_dims = input_dims
        self.modality_names = list(input_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.modality_transform = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.modality_transform[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
        self.dynamic_attention = DynamicCrossModalAttention(
            hidden_dim=hidden_dim,
            num_modalities=self.num_modalities,
            num_heads=num_heads,
            dropout=dropout
        )
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_modalities),
            nn.Sigmoid()
        )
        self.refinement_net = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim+hidden_dim*self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, modality_features):
        transformed_features = []
        for modality in self.modality_names:
            feat = self.modality_transform[modality](modality_features[modality])
            transformed_features.append(feat)
        
        stacked_features = torch.stack(transformed_features, dim=1)  # (batch, num_modalities, hidden_dim)
        attended_features,_ = self.dynamic_attention(stacked_features)
        attended_flat = attended_features.reshape(attended_features.size(0), -1)

        gate_weights = self.adaptive_gate(attended_flat)
        gated_features = attended_flat * gate_weights.repeat_interleave(self.hidden_dim, dim=1)
        refined_features = self.refinement_net(gated_features)
        combined_features = torch.cat([attended_features.reshape(attended_features.shape[0], -1)
        , refined_features], dim=1)  # (batch, hidden_dim * 2)
        
        output = self.output_proj(combined_features)
        
        return output


class ConceptAlignmentModule(nn.Module):
    def __init__(self, feature_dim, concept_dim=128, num_concepts=64):
        super(ConceptAlignmentModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        self.concept_prototypes = nn.Parameter(torch.randn(num_concepts, concept_dim))
        self.feature_to_concept = nn.Linear(feature_dim, concept_dim)
        self.concept_importance = nn.Linear(concept_dim, 1)
        
    def forward(self, features):
        concept_space = self.feature_to_concept(features)  # (batch, concept_dim)
        concept_similarities = F.cosine_similarity(
            concept_space.unsqueeze(1),  # (batch, 1, concept_dim)
            self.concept_prototypes.unsqueeze(0),  # (1, num_concepts, concept_dim)
            dim=-1
        )  # (batch, num_concepts)
        concept_activation = F.softmax(concept_similarities, dim=-1)  # (batch, num_concepts)
        concept_importance = torch.sigmoid(self.concept_importance(self.concept_prototypes))  # (num_concepts, 1)
        weighted_activations = concept_activation * concept_importance.squeeze(-1)  # (batch, num_concepts)
        
        return weighted_activations, concept_activation


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_features=1024, output_dim=64, dropout=0.1,embed = False,drug = False):
        super(SimpleCNN, self).__init__()
        self.em = embed
        self.one = nn.Sequential(nn.Conv1d(num_features, output_dim, 1),nn.ReLU(),nn.Dropout(dropout))
        self.three = nn.Sequential(nn.Conv1d(num_features, output_dim, 3,padding=1),nn.ReLU(),nn.Dropout(dropout))
        self.five = nn.Sequential(nn.Conv1d(num_features, output_dim, 5,padding=2),nn.ReLU(),nn.Dropout(dropout))
        self.embed = nn.Embedding(num_features, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.BatchNorm1d(output_dim)
        self.out = nn.Linear(output_dim*3, output_dim)
        self.pool = nn.AvgPool1d(output_dim)
        self.l1 = nn.Linear(1280,output_dim)
        self.re = nn.ReLU()
        self.o_t = nn.Sequential(nn.Conv1d(output_dim, output_dim, 3,padding=1),nn.ReLU(),nn.Dropout(dropout))
        self.o_f = nn.Sequential(nn.Conv1d(output_dim, output_dim, 5,padding=2),nn.ReLU(),nn.Dropout(dropout))
        self.t_f = nn.Sequential(nn.Conv1d(output_dim, output_dim, 5,padding=2),nn.ReLU(),nn.Dropout(dropout))
    def forward(self,seq):
        if self.em:
            sq = self.embed(seq)
        else:
            sq = self.re(self.l1(seq))
        o = self.o_f(self.o_t(self.one(sq)))
        t = self.t_f(self.three(sq))
        f = self.five(sq)
        x = torch.cat([o,t,f],dim=1)
        x = self.ln(x.permute(0,2,1))
        x = self.out(x)
        xp = self.pool(x).squeeze()
        return xp


class SCADDTA(nn.Module):
    def __init__(self, n_output=1, output_dim=256, dropout=0.1, num_features_mol=78, num_features_pro=34):
        super(SCADDTA, self).__init__()
        
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GATConv(num_features_mol*2, num_features_mol * 2)
        self.mol_conv3 = SuperGATConv(num_features_mol * 4, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 256)
        self.mol_fc_g2 = nn.Linear(256, output_dim)
        
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GATConv(num_features_pro*2, num_features_pro * 2)
        self.pro_conv3 = SuperGATConv(num_features_pro * 4, num_features_pro * 4)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 256)
        self.pro_fc_g2 = nn.Linear(256, output_dim)
        
        self.sm = SimpleCNN(embed=True, num_features=1024, output_dim=output_dim)
        self.se = SimpleCNN(embed=True, num_features=1024, output_dim=output_dim)
        self.q = SimpleCNN(num_features=1024, output_dim=output_dim)
        self.d = SimpleCNN(num_features=1024, output_dim=output_dim)
        
        self.f1 = nn.Linear(1024, output_dim)  # e3fp
        self.f2 = nn.Linear(315, output_dim)   # ergfp
        self.f3 = nn.Linear(881, output_dim)   # pubfp
        self.f4 = nn.Linear(166, output_dim)   # maccsfp
        
        fusion_input_dims = {
            'mol_graph': output_dim,
            'pro_graph': output_dim,
            'smiles_seq': output_dim,
            'protein_seq': output_dim,
            'smi_m': output_dim,
            'seq_m': output_dim,
            'e3fp': output_dim,
            'ergfp': output_dim,
            'pubfp': output_dim,
            'maccsfp': output_dim
        }
        
        self.hyperbolic_fusion = SphericalConstrainedProjectionFusion(
            input_dims=fusion_input_dims,
            hidden_dim=output_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.adaptive_fusion = AdaptiveMultiModalFusion(
            input_dims=fusion_input_dims,
            output_dim=output_dim,
            num_heads=4,
            dropout=dropout
        )
        
        self.concept_aligner = ConceptAlignmentModule(
            feature_dim=output_dim*2,
            concept_dim=256,
            num_concepts=64
        )
        
        self.fc1 = nn.Linear(output_dim*2 + 64, 1024)  
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
    
    def forward(self, smiles, sequence,e3fp, ergfp, pubfp, maccsfp, data_mol, data_pro, smi_m, seq_m):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x1 = self.mol_conv1(mol_x, mol_edge_index)
        x1 = self.relu(x1)
        
        x2 = self.mol_conv2(torch.cat([x1, mol_x], 1), mol_edge_index)
        x2 = self.relu(x2)
        
        x3 = self.mol_conv3(torch.cat([x1, mol_x, x2], 1), mol_edge_index)
        x3 = self.relu(x3)
        x3 = gep(x3, mol_batch)
        
        x = self.relu(self.mol_fc_g1(x3))
        x = self.dropout(x)
        mol_graph_feat = self.mol_fc_g2(x)
        mol_graph_feat = self.dropout(mol_graph_feat)
        
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch
        xt1 = self.pro_conv1(target_x, target_edge_index)
        xt1 = self.relu(xt1)
        
        xt2 = self.pro_conv2(torch.cat([xt1, target_x], 1), target_edge_index)
        xt2 = self.relu(xt2)
        
        xt3 = self.pro_conv3(torch.cat([xt1, target_x, xt2], 1), target_edge_index)
        xt3 = self.relu(xt3)
        xt3 = gep(xt3, target_batch)
        
        xt = self.relu(self.pro_fc_g1(xt3))
        xt = self.dropout(xt)
        pro_graph_feat = self.pro_fc_g2(xt)
        pro_graph_feat = self.dropout(pro_graph_feat)
        
        smiles_seq_feat = self.sm(smiles)
        protein_seq_feat = self.se(sequence)
        smi_m_feat = self.q(smi_m)
        seq_m_feat = self.d(seq_m)
        
        e3fp_feat = self.f1(e3fp)
        ergfp_feat = self.f2(ergfp)
        pubfp_feat = self.f3(pubfp)
        maccsfp_feat = self.f4(maccsfp)
        
        modality_features = {
            'mol_graph': mol_graph_feat,
            'pro_graph': pro_graph_feat,
            'smiles_seq': smiles_seq_feat,
            'protein_seq': protein_seq_feat,
            'smi_m': smi_m_feat,
            'seq_m': seq_m_feat,
            'e3fp': e3fp_feat,
            'ergfp': ergfp_feat,
            'pubfp': pubfp_feat,
            'maccsfp': maccsfp_feat
        }
        
        
        hyp_fused = self.hyperbolic_fusion(modality_features)
        
        adaptive_fused = self.adaptive_fusion(modality_features)
        
        combined_fused = torch.cat([hyp_fused, adaptive_fused], dim=1)
        
        concept_weights, concept_activation = self.concept_aligner(combined_fused)
        
        final_features = torch.cat([combined_fused, concept_weights], dim=1)
        
        xc = self.bn1(self.fc1(final_features))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        
        xc = self.bn2(self.fc2(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        
        out = self.out(xc)
        
        return out, concept_activation
