import os, tqdm
import numpy as np

root_path = '/Users/kelseyzhang/Desktop/emotionAnalyze/'

def AudioSplitorTool(video_path, save_path):
    if not os.path.exists(save_path):
        _cmd = "ffmpeg -i {} -vn -f wav -acodec pcm_s16le -ac 1 -ar 16000 {} -y > /dev/null 2>&1".format(
            video_path, save_path)
        os.system(_cmd)

def make_audio(video_folder):
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)]
    for video_file in tqdm(video_files):
        path_part = video_file.split('/')
        path_part[-2] = 'wav'
        path_part[-1] = path_part[-1].split('.')[0]  # 把.mp4去掉
        file_path = '/'.join(path_part)
        save_path = os.path.join(file_path + '.wav')
        AudioSplitorTool(video_file, save_path)


def get_audio_feature(videoIDs):
    video_folder = root_path + "MELD_RAW/train/train_splits/"
    # make_audio(video_folder)

    videoAudios = {}
    csv_path = root_path + 'MELD_RAW/train/feature/'  # 特征文本文件所在目录(opensmile提取）
    csv_path2 = root_path + 'MELD_RAW/test/feature/'
    csv_list = os.listdir(csv_path)

    for dia_ID in videoIDs.keys():
        features_list = []
        features_array = []
        for utt_ID in videoIDs[dia_ID]:
            if dia_ID>=1039:
                dia_ID_ = dia_ID - 1039
                file_path = os.path.join(csv_path2 + 'dia' + str(dia_ID_) + '_utt' + str(utt_ID) + '.csv')
            else:
                file_path = os.path.join(csv_path + 'dia' + str(dia_ID) + '_utt' + str(utt_ID) + '.csv')
            f = open(file_path)
            last_line = f.readlines()[-1]
            f.close()
            features = last_line.split(',')
            features = features[1:-1]
            features_ = [float(x) for x in features]
            features_list.append(features_)
        features_array = np.array(features_list)
        videoAudios[dia_ID] = features_array

    cols = []
    for i in videoAudios.keys():
        threshold = 1
        cols_to_delete = np.where(videoAudios[i].max(axis=0) > threshold)[0]
        cols.append(cols_to_delete)

    cc = max(cols,key=len)
    for i in videoAudios.keys():
        videoAudios[i] = np.delete(videoAudios[i], cc, axis=1)
    return videoAudios


