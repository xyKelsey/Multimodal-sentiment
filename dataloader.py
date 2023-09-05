import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size, valid, datafile):
    trainset = MELDDataset(datafile)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=trainset.collate_fn)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, collate_fn=trainset.collate_fn)

    testset = MELDDataset(datafile, train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn)

    return train_loader, valid_loader, test_loader

class MELDDataset(Dataset):
    '''
    MELD数据结构：
    - videoIDs: 一个dict, key是所有数据集中的Dialogue_ID, value是每个Dialogue_ID包含的所有的utterance ID
    - videoSpeakers: 一个dict, key是所有数据集中的Dialogue_ID, value是用one-hot encoding来表示的每一句话的说话人
    - videoLabels: 一个dict, key是所有数据集中的Dialogue_ID, value是每个Dialogue_ID包含的所有话的情绪标签
    - videoText: 采用Glove得到的文本特征向量
    - videoAudio: 采用OpenSMILE得到的音频特征向量
    - videoVisual: 采用Densenet提取得到的视觉特征向量
    - videoSentence: 所有数据集中每一句话的原始文本
    - trainVid: 训练集和验证集的所有Dialogue_ID
    - testVid: 测试集中的所有Dialogue_ID
    '''

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.LongTensor(self.videoLabels[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
