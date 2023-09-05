'''
使用OpenSmile提取音频特征，在'../opensmile/build/progsrc/smilextract'文件夹中运行
'''

import os
audio_path='/Users/kelseyzhang/Desktop/MELD_RAW/train/wav'
output_path='/Users/kelseyzhang/Desktop/MELD_RAW/train/feature'
audio_list=os.listdir(audio_path)
for audio in audio_list:
    if audio[-4:]=='.wav':
        this_path_input=os.path.join(audio_path,audio)
        this_path_output=os.path.join(output_path,audio[:-4]+'.csv')
        cmd='./SMILExtract -C /Users/kelseyzhang/opensmile/config/is09-13/IS10_paraling_compat.conf ' \
            '-I '+this_path_input+' -O '+this_path_output
        os.system(cmd)

