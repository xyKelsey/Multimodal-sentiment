import torch
import torch.nn as nn
import torch.nn.functional as F
from myGCN import my_GCN

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 1e-6

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        # label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)
        label_onehot = torch.zeros([seq_length, labels_length]).scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits, dim=-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

'''
将一个batch中的多个对话转换为一个图，其中每个句子对应一个节点，每个节点的特征由句子的特征组成
'''
def simple_batch_graphify(features, lengths):
    node_features = []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0) #一个batch里16个对话中所有句子特征拼接
    return node_features

class GCNModel(nn.Module):

    def __init__(self, text_feature, visual_feature, audio_feature, g_hidden_size, speakers_num,
                 classes, dropout, alpha):

        super(GCNModel, self).__init__()
        self.alpha = alpha
        self.dropout = dropout
        hidden_a, hidden_v, hidden_t = 300, 300, 300
        self.linear_a = nn.Linear(audio_feature, hidden_a) #300, 200
        self.linear_v = nn.Linear(visual_feature, hidden_v) #342, 200
        self.linear_t = nn.Linear(text_feature, hidden_t) #600,200


        self.graph_model = my_GCN(n_dim=300, nlayers=64, nhidden=g_hidden_size, dropout=self.dropout, lamda=0.5,
                                  alpha=0.1, variant=True, speakers_num=speakers_num)
        self.fc1 = nn.Linear(1500, classes)
        self.self_attention = nn.MultiheadAttention(300, num_heads=2)


    def forward(self, fea, speaker, utt_num, fea_a, fea_v, fea_t):

        fea_t = self.linear_t(fea_t)
        fea_a = self.linear_a(fea_a)
        fea_v = self.linear_v(fea_v)
        #
        # features = torch.cat([fea_a, fea_v, fea_t], dim=0)  # 将三种特征拼接在一起
        # attention_output, _ = self.self_attention(features, features, features)
        # fea_a = attention_output[:max(utt_num), :, :]
        # fea_v = attention_output[max(utt_num):max(utt_num)*2, :, :]
        # fea_t = attention_output[max(utt_num)*2:max(utt_num)*3, :, :]

        features_a = simple_batch_graphify(fea_a, utt_num) #seq_length,300
        features_v = simple_batch_graphify(fea_v, utt_num)
        features_t = simple_batch_graphify(fea_t, utt_num)

        x = self.graph_model(features_a, features_v, features_t, utt_num, speaker)

        x = F.dropout(x, self.dropout, training=self.training)

        x = nn.ReLU()(x)
        x = self.fc1(x)
        output = F.log_softmax(x, 1)
        return output
