# -*- coding:utf-8 -*-
# author: Xinge
# @file: dataset_coda.py 

import os
import numpy as np
from torch.utils import data
import yaml
import pickle
import re
import json

import torch
from torch.utils.data import DataLoader

from os.path import join

REGISTERED_dataset_CODa_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_dataset_CODa_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_dataset_CODa_CLASSES, f"exist class: {REGISTERED_dataset_CODa_CLASSES}"
    REGISTERED_dataset_CODa_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_dataset_CODa_CLASSES
    assert name in REGISTERED_dataset_CODa_CLASSES, f"available class: {REGISTERED_dataset_CODa_CLASSES}"
    return REGISTERED_dataset_CODa_CLASSES[name]

@register_dataset
class CODataset(data.Dataset):
    def __init__(
        self, 
        data_path, 
        imageset="train", 
        return_ref=False, 
        label_mapping="CODa_v2.yaml"
    ):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            CODayaml = yaml.safe_load(stream)
        self.learning_map = CODayaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path

        self.im_idx = []
        self._load_frame_list(data_path)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)
    
    def _load_frame_list(self, data_path):
        # Loop through all metadata files and build train, val, test list
        metadata_dir = join(data_path, "metadata")
        metadata_files = [file for file in os.listdir(metadata_dir) if file.endswith(".json")]
        
        imset_map = {
            "train": "training",
            "val": "validation",
            "test": "testing"
        }

        imageset = imset_map[self.imageset]
        for metadata_file in metadata_files:
            meta_path = join(metadata_dir, metadata_file)

            meta_dict = json.load(open(meta_path, 'r'))
            self.im_idx.extend(meta_dict['SemanticSegmentation'][imageset])
        
        self.im_idx = [join(data_path, subpath) for subpath in self.im_idx]
        
        def helper_filesplitter(subpath):
            fname = subpath.split('/')[-1]
            fname_pre = fname.split('.')[0]
            index = int(''.join(fname_pre.split('_')[-2:-1]))
            return index

        if imageset=="testing":
            # Sort test split for better image demos
            self.im_idx = sorted(self.im_idx, key=helper_filesplitter)

    def collate_fn(self, batch):
        # Stack images into a single tensor
        pc_list = [item[0] for item in batch]

        # Stack images into a single tensorbatch
        label_list = [item[1] for item in batch]

        if self.return_ref:
            ref_list = [item[2] for item in batch]
            ref = torch.stack(ref_list, dim=0)
            return (pc, label, ref)
        return (pc, label)

    def _load_pc(self, seq, frame):
        pc_path = join(
            self.data_path, 
            "3d_comp", 
            "os1", 
            seq, 
            f"3d_comp_os1_{seq}_{frame}.bin"
        )
        pc_np = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))

        return pc_np

    def _load_label(self, seq, frame):
        label_path = join(
            self.data_path, 
            "3d_semantic", 
            "os1", 
            seq, 
            f"3d_semantic_os1_{seq}_{frame}.bin"
        )

        #1 Load remapped label file
        with open(label_path, "rb") as annotated_file:
            label_np = np.array(list(annotated_file.read())).reshape((-1, 1))
        label_np = np.vectorize(self.learning_map.__getitem__)(label_np)

        return label_np.astype(np.uint8)

    def _get_frame_info(self, idx):
        filename = os.path.basename(self.im_idx[idx]).split('.')[0]
        seq, frame = filename.split('_')[-2:]
        return seq, frame

    def __getitem__(self, idx):
        seq, frame = self._get_frame_info(idx)

        # Load point cloud
        pc = self._load_pc(seq, frame)

        # Load label
        pc_annotation = self._load_label(seq, frame)

        if self.return_ref:
            data_tuple = (pc[:, :3], pc_annotation, pc[:, 3])
        else:    
            data_tuple = (pc[:, :3], pc_annotation)
        return data_tuple

if __name__ == '__main__':
    print("------   CODataset Test   -----")
    cfg_path = './config/coda.yaml'
    assert os.path.exists(cfg_path), f'Config file {cfg_path} does not exist'
    with open(cfg_path, 'r') as f:
        cfg_file = yaml.safe_load(f)

    #0 Initialize tracker instance and GroundedSAM
    sem_dataset_train = CODataset(
        data_path=cfg_file['train_data_loader']['data_path'],
        imageset=cfg_file['train_data_loader']['imageset'],
        return_ref=cfg_file['train_data_loader']['return_ref'],
        label_mapping=cfg_file['dataset_params']['label_mapping']
    )
    print("------   CODataset Initialized     ------")
    #2 Test basic loop through whole dataset
    
    dataloader = DataLoader(sem_dataset_train, batch_size=2, shuffle=False) #, collate_fn=sem_dataset_train.collate_fn)
    for i, batch in enumerate(dataloader):
        print(i, "pc ", batch[0].shape, " label ", batch[1].shape)

        # Break after the first batch to keep the output short
        if i == 0:
            break


def absoluteFilePaths(directory, CODa=False):
    print(directory)
    for dirpath, _, filenames in os.walk(directory):
        if not CODa:
            filenames.sort()
        else:
            filenames.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

# load Semantic KITTI class info
def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        remap_idx = semkittiyaml['learning_map'][i]
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['remap_labels'][remap_idx]

    return SemKITTI_label_name