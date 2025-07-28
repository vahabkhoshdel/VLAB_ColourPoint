import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py
from tqdm.auto import tqdm


class Dataset(Dataset):

    GRAVITATIONAL_AXIS = 1

    def __init__(self, path, split='train', scale_mode='shape_unit', transform=None):
        super().__init__()
        assert split in ('train', 'val', 'test')
        assert scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34', None)

        self.path = path
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):
        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f'stats_{self.split}.pt')

        if os.path.exists(stats_path):
            self.stats = torch.load(stats_path)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pcs = torch.from_numpy(f['pointcloud'][...])  # (B, N, 3)

        B, N, _ = pcs.size()
        mean = pcs.view(B * N, -1).mean(dim=0)
        std = pcs.view(-1).std()

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_path)
        return self.stats

    def load(self):
        with h5py.File(self.path, 'r') as f:
            pcs = f['pointcloud'][...]  # (B, N, 3)
            ids = [id.decode('utf-8') for id in f['id'][...]]
            shifts = f['shift'][...]
            scales = f['scale'][...]
            labels = f['label'][...]  # (B,)

        for i in range(len(pcs)):
            pc = torch.from_numpy(pcs[i])
            pc_id = ids[i]
            shift = torch.from_numpy(shifts[i].reshape(1, 3))
            scale = torch.from_numpy(scales[i].reshape(1, 1))
            label = int(labels[i]) 

            # Optional normalization override
            if self.scale_mode == 'global_unit':
                shift = pc.mean(dim=0, keepdim=True)
                scale = self.stats['std'].view(1, 1)
            elif self.scale_mode == 'shape_unit':
                shift = pc.mean(dim=0, keepdim=True)
                scale = pc.flatten().std().view(1, 1)
            elif self.scale_mode == 'shape_half':
                shift = pc.mean(dim=0, keepdim=True)
                scale = pc.flatten().std().view(1, 1) / 0.5
            elif self.scale_mode == 'shape_34':
                shift = pc.mean(dim=0, keepdim=True)
                scale = pc.flatten().std().view(1, 1) / 0.75
            elif self.scale_mode == 'shape_bbox':
                pc_max, _ = pc.max(dim=0, keepdim=True)
                pc_min, _ = pc.min(dim=0, keepdim=True)
                shift = (pc_min + pc_max) / 2
                scale = (pc_max - pc_min).max().view(1, 1) / 2

            pc = (pc - shift) / scale

            self.pointclouds.append({
                'pointcloud': pc,
                'id': pc_id,
                'shift': shift,
                'scale': scale,
                'label': label
            })

        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k: v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == "__main__":
    train_dset = Dataset(path="../data/densenet.hdf5", split='train', scale_mode='shape_unit')
    train_loader = DataLoader(train_dset, batch_size=12, num_workers=0)

    print(train_dset)

    for batch in train_loader:
        print(batch)
        break
        