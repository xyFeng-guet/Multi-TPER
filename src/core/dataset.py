import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class create_Dataset():
    def __init__(self, datasetNmae, labelType):
        DATA_MAP = {
            'emotake': self.__init_emotake,
            'other': None
        }
        data, label = DATA_MAP[datasetNmae](labelType)
        self.d_l = {'data': data, 'label': label}

    def __init_emotake(self, labelType):
        au_data = np.load('data/aus.npy').transpose(0, 2, 1)
        em_data = np.load('data/ems.npy').transpose(0, 2, 1)
        hp_data = np.load('data/hps.npy').transpose(0, 2, 1)

        bp_data = np.load('data/bps.npy')
        bp_data = bp_data.reshape(bp_data.shape[0], bp_data.shape[1], -1).transpose(0, 2, 1)

        combine_data = {'au': au_data, 'em': em_data, 'hp': hp_data, 'bp': bp_data}

        quality_label = np.load('data/quality.npy', allow_pickle=True)
        quality_label = np.array(quality_label, dtype=int)
        ra_label = np.load('data/ra.npy', allow_pickle=True)
        ra_label = np.array(ra_label, dtype=int)
        readiness_label = np.load('data/readiness.npy', allow_pickle=True)
        readiness_label = np.array(readiness_label, dtype=int)

        if labelType == 'quality':
            combine_label = quality_label
        elif labelType == 'ra':
            combine_label = ra_label
        else:
            combine_label = readiness_label

        return combine_data, combine_label

    def __len__(self):
        return len(self.dataset['label'])


def create_DataLoader(opt, dataset):
    dataset = MDDataset(dataset)
    dataLoader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True
    )
    return dataLoader


class MDDataset(Dataset):
    def __init__(self, dataset):
        self.reconstruct_dataset(dataset)

    def reconstruct_dataset(self, dataset):
        self.au = dataset['au']
        self.em = dataset['em']
        self.hp = dataset['hp']
        self.bp = dataset['bp']

        self.au_lengths = len(dataset['au'])
        self.em_lengths = len(dataset['em'])
        self.hp_lengths = len(dataset['hp'])
        self.bp_lengths = len(dataset['bp'])

        # # Clear dirty data
        # self.audio[self.audio == -np.inf] = 0
        # self.vision[self.vision == -np.inf] = 0

        self.label = dataset[self.mode]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = {
            'au': self.rawText[index],
            'em': torch.Tensor(self.text[index]),
            'hp': torch.Tensor(self.audio[index]),
            'bp': torch.Tensor(self.vision[index]),
            'au_lengths': self.audio_lengths[index],
            'em_lengths': self.vision_lengths[index],
            'hp_lengths': self.audio_lengths[index],
            'bp_lengths': self.vision_lengths[index],
            'label': self.label[index],
            'index': index
        }
        return sample
