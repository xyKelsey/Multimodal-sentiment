import os
import pickle
from io import FileIO
from typing import Any, Tuple

import h5py
import numpy as np
from overrides import overrides
import torch
import torch.nn
import torch.utils.data
import torchvision
from tqdm import tqdm

from visual.c3d import C3D
from visual.i3d import I3D
from visual.dataset import FrameDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps")

root_path = '/Users/kelseyzhang/Desktop/emotionAnalyze/'

def Video2FrameTOOL(video_path, frame_dir):
    basename = os.path.basename(video_path)
    if os.path.isfile(video_path):
        basename = basename[:basename.rfind('.')]
    save_dir = os.path.join(frame_dir, basename)
    if not (os.path.exists(save_dir)):
        os.mkdir(save_dir)
        # cmd = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}/'.format(video_path, fps, save_dir) + '%4d.jpg'
        cmd = 'ffmpeg -i {} -threads 5 -f image2 {}/'.format(video_path, save_dir) + '%4d.jpg'
        os.system(cmd)
    return save_dir

def Video2Frame(video_path):
    frame_dir = os.path.join(root_path, 'frames/')
    videos = os.listdir(root_path + video_path)
    for video in videos:
        get_frame = Video2FrameTOOL(video_path + video, frame_dir)

def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    resnet152.eval()
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152

def pretrained_c3d() -> torch.nn.Module:
    c3d = C3D(pretrained=True)
    c3d.eval()
    for param in c3d.parameters():
        param.requires_grad = False
    return c3d

def pretrained_i3d() -> torch.nn.Module:
    i3d = I3D(pretrained=True)
    i3d.eval()
    for param in i3d.parameters():
        param.requires_grad = False
    return i3d

def save_resnet_features() -> None:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = FrameDataset(transform=transforms)

    resnet = pretrained_resnet152().to(DEVICE)

    class Identity(torch.nn.Module):
        @overrides
        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            return input_

    resnet.fc = Identity()  # avoid computing the fc1000 layer

    with h5py.File(FrameDataset.features_file_path("resnet", "res5c"), "w") as res5c_features_file, \
            h5py.File(FrameDataset.features_file_path("resnet", "pool5"), "w") as pool5_features_file:

        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            res5c_features_file.create_dataset(video_id, shape=(video_frame_count, 2048, 7, 7))
            pool5_features_file.create_dataset(video_id, shape=(video_frame_count, 2048))

        res5c_output = None

        def avg_pool_hook(_module: torch.nn.Module, input_: Tuple[torch.Tensor], _output: Any) -> None:
            nonlocal res5c_output
            res5c_output = input_[0]

        resnet.avgpool.register_forward_hook(avg_pool_hook)

        total_frame_count = sum(dataset.frame_count_by_video_id[video_id] for video_id in dataset.video_ids)
        with tqdm(total=total_frame_count, desc="Extracting ResNet features") as progress_bar:
            for instance in torch.utils.data.DataLoader(dataset):
                video_id = instance["id"][0]
                frames = instance["frames"][0].to(DEVICE)

                batch_size = 128
                for start_index in range(0, len(frames), batch_size):
                    end_index = min(start_index + batch_size, len(frames))
                    frame_ids_range = range(start_index, end_index)
                    frame_batch = frames[frame_ids_range]

                    avg_pool_value = resnet(frame_batch)
                    for x, y in zip(frame_ids_range, range(0, avg_pool_value.shape[0])):
                        res5c_features_file[video_id][x] = res5c_output.cpu()[y]
                        pool5_features_file[video_id][x] = avg_pool_value.cpu()[y]
                        # with open('xxxx.pkl', 'wb') as f:
                        #     pickle.dump(avg_pool_value.cpu()[y], f)
                    # res5c_features_file[video_id][frame_ids_range] = res5c_output.cpu()  # noqa
                    # pool5_features_file[video_id][frame_ids_range] = avg_pool_value.cpu()

                    progress_bar.update(len(frame_ids_range))


def save_c3d_features() -> None:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(128),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = FrameDataset(transform=transforms)

    c3d = pretrained_c3d().to(DEVICE)

    with h5py.File(FrameDataset.features_file_path("c3d", "fc7"), "w") as fc7_features_file:
        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            # feature_count = video_frame_count - 16 + 1
            feature_count = video_frame_count - 16 + 1 if video_frame_count - 15 > 0 else video_frame_count
            fc7_features_file.create_dataset(video_id, shape=(feature_count, 4096))

        for instance in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting C3D features"):
            video_id = instance["id"][0]  # noqa
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            feature_count = video_frame_count - 16 + 1
            frames = instance["frames"][0].to(DEVICE)
            frames = frames.unsqueeze(0)  # Add batch dimension
            frames = frames.transpose(1, 2)  # C3D expects (B, C, T, H, W)

            for i in range(feature_count):
                output = c3d.extract_features(frames[:, :, i:i + 16, :, :]).squeeze()
                fc7_features_file[video_id][i, :] = output.cpu().data.numpy()


def save_i3d_features():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])
    dataset = FrameDataset(transform=transforms)

    i3d = pretrained_i3d().to(DEVICE)

    with h5py.File(FrameDataset.features_file_path("i3d", "avg_pool"), "w") as avg_pool_features_file:
        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            feature_count = video_frame_count - 16 + 1 if video_frame_count-15 > 0 else video_frame_count
            avg_pool_features_file.create_dataset(video_id, shape=(feature_count, 1024))
        videoVisual = {}
        for instance in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting I3D features"):
            video_id = instance["id"][0]  # noqa
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            feature_count = video_frame_count - 16 + 1
            frames = instance["frames"][0].to(DEVICE)
            frames = frames.unsqueeze(0)  # Add batch dimension
            frames = frames.transpose(1, 2)  # I3D expects (B, C, T, H, W)
            visualF = []
            for i in range(feature_count):
                output = i3d.extract_features(frames[:, :, i:i + 16, :, :]).squeeze()
                visualF.append(output.cpu().data.numpy())
                avg_pool_features_file[video_id][i, :] = output.cpu().data.numpy()
            visualF = np.array(visualF)
            dia_ID = int(video_id[3:4])
            if dia_ID in videoVisual:
                videoVisual[dia_ID] = np.vstack([videoVisual[dia_ID], np.average(visualF, axis=0)])
            else:
                videoVisual[dia_ID] = np.average(visualF, axis=0)
                videoVisual[dia_ID] = videoVisual[dia_ID].reshape(1, videoVisual[dia_ID].shape[0])
    return videoVisual

# def save_i3d_features():
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(256),
#         torchvision.transforms.CenterCrop(224),
#         torchvision.transforms.ToTensor(),
#     ])
#     dataset = FrameDataset(transform=transforms)
#
#     i3d = pretrained_i3d().to(DEVICE)
#
#     for video_id in dataset.video_ids:
#         video_frame_count = dataset.frame_count_by_video_id[video_id]
#         # feature_count = video_frame_count - 16 + 1 if video_frame_count-15 > 0 else video_frame_count
#         feature_count = video_frame_count
#     videoVisual = {}
#     for instance in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting I3D features"):
#         video_id = instance["id"][0]  # noqa
#         video_frame_count = dataset.frame_count_by_video_id[video_id]
#         feature_count = video_frame_count - 16 + 1
#         frames = instance["frames"][0].to(DEVICE)
#         frames = frames.unsqueeze(0)  # Add batch dimension
#         frames = frames.transpose(1, 2)  # I3D expects (B, C, T, H, W)
#         visualF = []
#         for i in range(feature_count):
#             output = i3d.extract_features(frames[:, :, i:i + 16, :, :]).squeeze()
#             visualF.append(output.cpu().data.numpy())
#             # avg_pool_features_file[video_id][i, :] = output.cpu().data.numpy()
#         visualF = np.array(visualF)
#         dia_ID = video_id[3:4]
#         utt_ID = video_id[-1]
#
#         if dia_ID in videoVisual:
#             videoVisual[dia_ID] = np.vstack([videoVisual[dia_ID], np.average(visualF, axis=0)])
#         else:
#             videoVisual[dia_ID] = np.average(visualF, axis=0)
#             videoVisual[dia_ID] = videoVisual[dia_ID].reshape(1, videoVisual[dia_ID].shape[0])
#
#     return videoVisual

def get_visual_feature():
    network = 'i3d'
    videoVisual = {}
    if network == "resnet":
        save_resnet_features()
    elif network == "c3d":
        save_c3d_features()
    elif network == "i3d":
        videoVisual = save_i3d_features()
    else:
        raise ValueError(f"Network type not supported: {network}")
    return videoVisual

if __name__ == "__main__":
    video_path = 'MELD_RAW/train/train_splits/'
    # Video2Frame(video_path)
    videoVisual = get_visual_feature()

    file = os.path.join('visual_feature.pkl')
    if isinstance(file, str):
        os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
        file = open(file, 'wb')
        pickle.dump(videoVisual, file)
        file.close()
    elif isinstance(file, FileIO):
        pickle.dump(videoVisual, file)
