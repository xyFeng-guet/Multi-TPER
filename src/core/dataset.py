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

        combine_data = [{'au': au_data[i], 'em': em_data[i], 'hp': hp_data[i], 'bp': bp_data[i]} for i in range(len(au_data))]

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
        data, label = dataset[0], dataset[1]
        au, em, hp, bp = [], [], [], []
        for i in range(len(data)):
            au.append(data[i]['au'])
            em.append(data[i]['em'])
            hp.append(data[i]['hp'])
            bp.append(data[i]['bp'])

        self.au = torch.tensor(np.array(au), dtype=torch.float32)
        self.em = torch.tensor(np.array(em), dtype=torch.float32)
        self.hp = torch.tensor(np.array(hp), dtype=torch.float32)
        self.bp = torch.tensor(np.array(bp), dtype=torch.float32)

        self.au_lengths = len(self.au[0])
        self.em_lengths = len(self.em[0])
        self.hp_lengths = len(self.hp[0])
        self.bp_lengths = len(self.bp[0])

        # # Clear dirty data
        # self.audio[self.audio == -np.inf] = 0
        # self.vision[self.vision == -np.inf] = 0

        self.label = torch.tensor(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = {
            'au': self.au[index],
            'em': self.em[index],
            'hp': self.hp[index],
            'bp': self.bp[index],
            'au_lengths': self.au_lengths,
            'em_lengths': self.em_lengths,
            'hp_lengths': self.hp_lengths,
            'bp_lengths': self.bp_lengths,
            'label': self.label[index],
            'index': index
        }
        return sample
