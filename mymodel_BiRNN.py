import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np, random

'''
实现全局注意力机制（Global Attention）
'''
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

        M_ = M.permute(1, 2, 0)  # batch_size, global_size, seq_length
        x_ = self.transform(x).unsqueeze(1)  # batch_size, 1, global_dimension
        alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch_size, 1, seq_length
        attention_pool = torch.bmm(alpha, M.transpose(0, 1))
        attention_pool = attention_pool[:, 0, :]  # batch_size, global_size

        return attention_pool

'''
Multimodal Recurrent Neural Network的基本单元 
'''
class RNNCell(nn.Module):

    def __init__(self, all_feature_num, global_size, speaker_size, emotion_size, dropout=0.1):
        super(RNNCell, self).__init__()

        self.all_feature_num = all_feature_num
        self.global_size = global_size
        self.speaker_size = speaker_size
        self.emotion_size = emotion_size

        self.global_cell = nn.GRUCell(all_feature_num + speaker_size, global_size)  # 全局GRU
        self.party_cell = nn.GRUCell(all_feature_num + global_size, speaker_size)  # 参与者GRU
        self.emotion_cell = nn.GRUCell(speaker_size, emotion_size)  # 情感GRU
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(global_size, all_feature_num)

    ''' 
    返回当前对话所包含的说话人 
    '''
    def select_parties(self, X, indices):
        speaker0_sel = []
        for idx, x in zip(indices, X):
            speaker0_sel.append(x[idx].unsqueeze(0))
        speaker0_sel = torch.cat(speaker0_sel, 0)
        return speaker0_sel

    def forward(self, feature_, speaker_mask, global_history, speaker0, emotion0):
        '''
        feature_ -> batch_size, all_feature_num
        speaker_mask -> batch_size, participant
        global_history -> t-1, batch_size, global_size
        speaker0 -> batch_size, participant, speaker_size
        emotion0 -> batch_size, emotion_size
        '''
        # 返回speaker_mask中每一行的最大值对应的下标
        speaker_mask_id = torch.argmax(speaker_mask, 1)
        speaker0_sel = self.select_parties(speaker0, speaker_mask_id)

        # 状态信息第一次都是用全零初始化
        # 全局GRU input：[该时刻特征feature_+当前话语speaker0_sel]+前一个全局GRU状态global_history_
        feature_speak = torch.cat([feature_, speaker0_sel], dim=1)
        global_history_ = torch.zeros(feature_.size()[0], self.global_size) if global_history.size()[0] == 0 else global_history[-1]
        global_state = self.global_cell(feature_speak, global_history_) #batch_size, global_size
        global_state = self.dropout(global_state)

        # 根据上一个时间的状态global_history和对话内容特征feature_，通过attention机制，得到上下文向量context
        if global_history.size()[0] == 0:
            context = torch.zeros(feature_.size()[0], self.global_size) #batch_size global_size
        else:
            context = self.attention(global_history, feature_)

        # 参与者GRU：input: [历史说话信息(经过attention)context+当前特征feature_]+前一刻说话特征speaker0 output:当前时刻说话人的信息parties_state
        # 将第二维扩展为speaker_mask.size()[1]
        feature_context = torch.cat([feature_, context], dim=1).unsqueeze(1).expand(-1, speaker_mask.size()[1], -1) #batch_size, onehot_len, feature_size+global_size
        feature_context = feature_context.contiguous().view(-1, self.all_feature_num + self.global_size) #batch_size*9, fea_size+g_size
        parties_state = self.party_cell(feature_context, speaker0.view(-1, self.speaker_size)) #batch_size*9, speaker_size
        parties_state = parties_state.view(feature_.size()[0], -1, self.speaker_size) #batch_size, onehot_len, speaker_size
        parties_state = self.dropout(parties_state)

        # 其他参与者的状态更新
        listener0 = speaker0 #倾听者状态不变
        speaker_mask_ = speaker_mask.unsqueeze(2)
        speaker = listener0 * (1 - speaker_mask_) + parties_state * speaker_mask_

        # 情绪GRU：input：前一刻情绪分析GRU的隐藏状态emotion0+当前说话人的状态（当前时刻话语的表示）parties
        emotion0 = torch.zeros(speaker_mask.size()[0], self.emotion_size) if emotion0.size()[0] == 0 else emotion0 #batch_Size, emotion_size
        parties = self.select_parties(speaker, speaker_mask_id)
        emotion = self.emotion_cell(parties, emotion0) #batch_Size, emotion_size
        emotion = self.dropout(emotion)

        return emotion, global_state, speaker


'''
由基本单元构成的Multimodal Conversational Recurrent Neural Network  
'''
class myRNN(nn.Module):

    def __init__(self, all_feature_num, global_size, speaker_size, emotion_size, dropout=0.1):
        super(myRNN, self).__init__()

        self.all_feature_num = all_feature_num
        self.global_size = global_size
        self.speaker_size = speaker_size
        self.emotion_size = emotion_size
        self.dropout = nn.Dropout(dropout)

        self.conversational_cell = RNNCell(all_feature_num, global_size, speaker_size, emotion_size, dropout)

    def forward(self, feature_concat, speaker_mask):
        '''
        feature_concat -> sequence_length, batch_size, all_feature_num
        speaker_mask -> sequence_length, batch_size, party
        '''
        global_history = torch.zeros(0) # 0-dimensional tensor
        speaker0 = torch.zeros(speaker_mask.size()[1], speaker_mask.size()[2], self.speaker_size)  # batch_size, onehot_len, speaker_size
        emotion0 = torch.zeros(0)  # batch_size, emotion_size
        emotion = emotion0
        speaker_ = speaker0
        for feature_, speaker_mask_ in zip(feature_concat, speaker_mask): #遍历每句话所对应的特征
            sentiment_, global_, speaker_ = self.conversational_cell(feature_, speaker_mask_, global_history, speaker_, emotion0)
            global_history = torch.cat([global_history, global_.unsqueeze(0)], 0)
            emotion = torch.cat([emotion, sentiment_.unsqueeze(0)], 0)

        return emotion  # sequence_length, batch_size, emotion_size


'''
由Multimodal Conversational Recurrent Neural Network构成的主模型
'''
class RNNModel(nn.Module):

    def __init__(self, all_feature_num, global_size, speaker_size, emotion_size, hidden_size, num_classes,
                 dropout_rec=0.1, dropout=0.1):
        super(RNNModel, self).__init__()

        self.all_feature_num = all_feature_num
        self.global_size = global_size
        self.speaker_size = speaker_size
        self.D_sentiment = emotion_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout + 0.15)

        self.conversational_rnn = myRNN(all_feature_num, global_size, speaker_size, emotion_size, dropout_rec)
        self.linear = nn.Linear(emotion_size, hidden_size)
        self.softmax = nn.Linear(hidden_size, num_classes)
        self.attention = Attention(emotion_size, emotion_size)

    def forward(self, feature_concat, speaker_mask, utt_num=None, set_attention=False):
        '''
        feature_concat -> sequence_length, batch_size, all_feature_num
        #speaker_mask -> sequence_length, batch_size, party
        '''
        # speaker_mask = speaker_mask.transpose(0,1)
        emotions, alpha_forward = self.conversational_rnn(feature_concat, speaker_mask)  # sequence_length, batch, emotion_size
        emotions = self.dropout_rec(emotions)

        if set_attention:
            att_emotions = []
            for e in emotions:
                att_emotions.append(self.attention(emotions, e, mask=utt_num)[0].unsqueeze(0))
            att_emotions = torch.cat(att_emotions, dim=0)
            # relu: 线性整流函数
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        # log_softmax: mathematically equivalent to log(softmax(x))
        log_prob = F.log_softmax(self.softmax(hidden), 2)  # sequence_length, batch_size, num_classes

        return log_prob


'''
由两个Multimodal Conversational Recurrent Neural Network构成的双向主模型 
'''

class BiRNNModel(nn.Module):

    def __init__(self, all_feature_num, global_size, speaker_size, emotion_size, hidden_size, num_classes,
                 dropout_rec=0.1, dropout=0.1):
        super(BiRNNModel, self).__init__()

        self.all_feature_num = all_feature_num
        self.global_size = global_size
        self.speaker_size = speaker_size
        self.emotion_size = emotion_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout + 0.15)

        self.RNN_forward = myRNN(all_feature_num, global_size, speaker_size, emotion_size, dropout_rec)
        self.RNN_backward = myRNN(all_feature_num, global_size, speaker_size, emotion_size, dropout_rec)

        self.fc1 = nn.Linear(emotion_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)
        self.attention = Attention(emotion_size * 2, emotion_size * 2)

    def reverse_sequence(self, X, utt_num):
        '''
        X -> sequence_length, batch_size, dimension
        mask -> batch_size, sequence_length
        '''
        X_ = X.transpose(0, 1)
        # mask_sum = torch.sum(mask, 1).int()
        xfs = []
        for x, c in zip(X_, utt_num):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, feature_concat, speaker_mask, utt_num, set_attention=True):
        '''
        feature_concat -> sequence_length, batch_size, all_feature_num
        speaker_mask -> sequence_length, batch_size, party
        '''

        emotion_forward = self.RNN_forward(feature_concat, speaker_mask)  # sequence_length, batch_size, emotion_size
        emotion_forward = self.dropout_rec(emotion_forward)
        reverse_feature = self.reverse_sequence(feature_concat, utt_num)
        reverse_speaker_mask = self.reverse_sequence(speaker_mask, utt_num)

        emotion_backward = self.RNN_backward(reverse_feature, reverse_speaker_mask)
        emotion_backward = self.reverse_sequence(emotion_backward, utt_num)
        emotion_backward = self.dropout_rec(emotion_backward)
        emotions = torch.cat([emotion_backward, emotion_forward], dim=-1)

        if set_attention:
            att_emotions = []
            for e in emotions:
                att_sen = self.attention(emotions, e)
                att_emotions.append(att_sen.unsqueeze(0))
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.fc1(att_emotions))
        else:
            hidden = F.relu(self.fc1(emotions))

        hidden = self.dropout(hidden)
        output = F.log_softmax(self.fc2(hidden), 2)  # sequence_length, batch_size, num_classes

        return output

'''
计算负对数似然损失 (Negative Log Likelihood Loss, NLLLoss)
'''
class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        # torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        # reduction (string, optional) – Specifies the reduction to apply to the output: 'sum': the output will be summed.
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, predict, target, mask):
        '''
        predict -> batch_size * sequence_length, num_classes
        target -> batch_size * sequence_length
        mask -> batch_size, sequence_length
        '''
        mask_ = mask.view(-1, 1)  # batch_size * sequence_length, 1
        if type(self.weight) == type(None):
            loss = self.loss(predict * mask_, target) / torch.sum(mask)
        else:
            # torch.squeeze(input, dim=None, out=None) → Tensor
            # Returns a tensor with all the dimensions of input of size 1 removed.
            # For example, if input is of shape: (A×1×B×C×1×D) then the out tensor will be of shape: (A×B×C×D) .
            loss = self.loss(predict * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())

        return loss



