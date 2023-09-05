import os
import pandas as pd
import pickle
from io import FileIO

from feature_extractor.audio_extractor import get_audio_feature
from text_extractor import get_text_feature
from visual_extractor import get_visual_feature

root_path = '/Users/kelseyzhang/Desktop/emotionAnalyze/'

train_path = root_path + 'MELD_RAW/train/train_splits'
train_list = os.listdir(train_path)

test_path = root_path + 'MELD_RAW/test/output_repeated_splits_test'
test_list = os.listdir(test_path)

train_label_path = root_path + 'MELD_RAW/label/train_sent_emo_1.csv'

train_label_data = pd.read_csv(train_label_path)

def dump_pkl(obj, file, make_path=True):
    if isinstance(file, str):
        if make_path:
            os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
        file = open(file, 'wb')
        pickle.dump(obj, file)
        file.close()
    elif isinstance(file, FileIO):
        pickle.dump(obj, file)
    else:
        raise NotImplementedError()

def get_video_basic2():
    videoIDs = {}
    videoSentences = {}
    videoSpeakers = {}
    videoLabels = {}
    label_index = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    df = train_label_data
    for _, row in df.iterrows():
        dia_num = int(row['Dialogue_ID'])
        utt_num = int(row['Utterance_ID'])
        label = row['Emotion']
        if dia_num in videoIDs:
            videoIDs[dia_num].append(utt_num)
            videoSentences[dia_num].append(row[1])
            videoSpeakers[dia_num].append(row[2])
            videoLabels[dia_num].append(label_index[label])
        else:
            videoIDs[dia_num] = [utt_num]
            videoSentences[dia_num] = [row[1]]
            videoSpeakers[dia_num] = [row[2]]
            videoLabels[dia_num] = [label_index[label]]

    for x in videoSpeakers.keys():
        s = pd.get_dummies(videoSpeakers[x])
        speaker_oh = s.values.tolist()
        for i in range(len(speaker_oh)):
            speaker_oh[i] = speaker_oh[i] + [0] * (9 - len(speaker_oh[i]))
        videoSpeakers[x] = speaker_oh
    return videoIDs, videoSentences, videoSpeakers, videoLabels


if __name__=='__main__':
    videoIDs, videoSentences, videoSpeakers, videoLabels = get_video_basic2() #dict
    videoAudios = get_audio_feature(videoIDs) #dict
    videoText = get_text_feature(videoSentences)
    videoVisual = get_visual_feature()

    MELD_feature = [videoIDs, videoSpeakers, videoLabels, videoText, videoAudios,
                    videoVisual, videoSentences]
    pklFile = os.path.join(f'MELD_feature.pkl')
    dump_pkl(MELD_feature, pklFile)
