import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class Attention(nn.Module):

    def __init__(self, global_size, feature_size):
        super(Attention, self).__init__()
        self.transform = nn.Linear(feature_size, global_size, bias=False)

    def forward(self, M, x):
        '''
        - M: (seq_length, batch_size, feature_size) 序列数据
        - x: (batch_size, feature_size) 全局特征
        alpha 注意力权重
        '''

        M_ = M.permute(1, 0)  # global_size, seq_length
        x_ = self.transform(x).unsqueeze(1)  # batch_size, global_dimension
        alpha = F.softmax(torch.bmm(x_, M_), dim=1)  # batch_size, seq_length
        attention_pool = torch.bmm(alpha, M.transpose(0, 1))
        attention_pool = attention_pool[:, 0, :]  # batch_size, global_size

        return attention_pool

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = nn.Parameter(torch.FloatTensor(self.in_features,self.out_features)) #权重矩阵为可训练参数
        self.reset_parameters()

    def reset_parameters(self): #初始化参数
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv) #使用均匀分布U(-stdv,stdv)初始化权重Tensor

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input) #稀疏矩阵乘法，得到A*H^l hi当前节点的特征表示
        if self.variant:
            support = torch.cat([hi,h0],1) #当前节点特征与初始特征合并
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual: #残差连接
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))

        # self.self_attn = nn.MultiheadAttention(200, num_heads=4)
        self.fc = nn.Linear(nfeat, nhidden)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training) #,300
        layer_inner = self.relu(self.fc(x))
        h0 = layer_inner

        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.relu(con(layer_inner, adj, h0, self.lamda, self.alpha, i+1))

        # layer_inner = layer_inner.transpose(0, 1)  # 调整维度顺序
        # attn_output, _ = self.self_attn(layer_inner, layer_inner, layer_inner)
        # layer_inner = attn_output.transpose(0, 1)

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training) #3*seq_length,200
        layer_inner = torch.cat([x, layer_inner], dim=-1) #3*seq_length,500
        return layer_inner

class MultiModalEncoder(nn.Module):
    def __init__(self, in_dims, n_modalities, hidden_dim, num_heads, dropout):
        super(MultiModalEncoder, self).__init__()

        self.in_dims = in_dims
        self.n_modalities = n_modalities
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # 定义每个模态的编码器
        self.modal_encoders = nn.ModuleList()
        for i in range(n_modalities):
            self.modal_encoders.append(nn.Linear(in_dims[i], hidden_dim))

        # 定义自注意力机制
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads)

    def forward(self, x):
        # 编码每个模态的特征
        modal_features = []
        for i in range(self.n_modalities):
            modal_feature = self.modal_encoders[i](x[i])
            # modal_feature = modal_feature.transpose(0, 1)  # 将维度顺序变为 (seq_len, batch_size, hidden_dim)
            modal_features.append(modal_feature)

        # 使用自注意力机制编码多模态特征
        modal_features = torch.cat(modal_features, dim=0)  # 将不同模态的特征堆叠在一起
        attn_output, _ = self.self_attn(modal_features, modal_features, modal_features)
        # attn_output = attn_output.transpose(0, 1)  # 将维度顺序变为 (batch_size, seq_len, hidden_dim)
        # attn_output = attn_output.mean(dim=1)  # 对序列长度取平均，得到多模态特征表示

        return attn_output

class my_GCN(nn.Module):
    def __init__(self, n_dim, nlayers, nhidden, dropout, lamda, alpha, variant, speakers_num):
        super(my_GCN, self).__init__()
        self.speaker_embeddings = nn.Embedding(speakers_num, n_dim)  # 9,200
        self.encoder = MultiModalEncoder([300,342,600], 3, nhidden, 2, 0.4)
        #200,64,100,7,variant=True
        self.graph_net = GCNII(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden,
                               dropout=dropout, lamda=lamda, alpha=alpha, variant=variant)

    def forward(self, a, v, t, utt_num, speaker):
        #speaker.shape:(max(dia_len),batch,onehot_len)  -->  (seq_length,onehot_len)
        speaker = torch.cat([speaker[:x,i,:] for i,x in enumerate(utt_num)],dim=0)
        spk_idx = torch.argmax(speaker, dim=-1) #找到当前行对应的说话人编号(每行中最大值所在的列索引)
        # spk_idx表示说话人编号，是一个一维张量，长度等于问题中对话的总条数。其中，spk_idx[i]表示第i条对话中的说话人编号。
        spk_emb_vector = self.speaker_embeddings(spk_idx) #seq_length,300
        t += spk_emb_vector
        adj = self.create_big_adj(a, v, t, utt_num) #求邻接矩阵A 3*seq_length,3*seq_length

        features = torch.cat([a, v, t], dim=0) #3*seq_length,300
        # features = self.encoder([a, v, t])
        # adj = self.create_big_adj(features, utt_num)
        features = self.graph_net(features, adj) #3*seq_length,500
        lens = sum(utt_num)
        features = torch.cat([features[:lens], features[lens: lens*2], features[lens*2: lens*3]], dim=-1) #avt三个特征合并
        return features
        # return features[:lens], features[lens: lens*2], features[lens*2: lens*3]

    def create_big_adj(self, a, v, t, dia_len):
        modal_num = 3
        all_length = t.shape[0] if len(t)!=0 else a.shape[0] if len(a) != 0 else v.shape[0]
        adj = torch.zeros((modal_num*all_length, modal_num*all_length))
        features = [a, v, t]

        start = 0
        for i in range(len(dia_len)):
            sub_adjs = []
            for j, x in enumerate(features): #相同模态边权重A
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                    temp = x[start:start + dia_len[i]] #一个dia中的特征
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1)) #对每个行向量中所有特征值的平方进行求和再开方->行向量的二范数
                    norm_temp = (temp.permute(1, 0) / vec_length) #对temp转置除以二范数->对行向量归一化转换为单位向量
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)  #余弦相似度矩阵 # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999
                    sim_matrix = 1 - torch.acos(cos_sim_matrix)/np.pi
                    sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                sub_adjs.append(sub_adj)
            dia_idx = np.array(np.diag_indices(dia_len[i]))
            for m in range(modal_num):
                for n in range(modal_num):
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n:
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adjs[m]
                    else: #同一对话的不同模态之间边的权重A
                        modal1 = features[m][start:start+dia_len[i]] #length, dim
                        modal2 = features[n][start:start+dia_len[i]]
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #length
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = (1 - torch.acos(dia_cos_sim)/np.pi)
                        idx = dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        adj[idx] = dia_sim

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5)) #度矩阵，每列求和写在对角线上
        adj = D.mm(adj).mm(D) #对A～进行归一化

        return adj

