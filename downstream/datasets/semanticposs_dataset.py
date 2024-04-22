import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import logging
from pathlib import Path


# This file is covered by the LICENSE file in the root of this project.
# original labels
labels = {
    0: "unlabeled",
    4: "1 person",
    5: "2+ person",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1",  # standing sign
    11: "traffic sign 2",  # hanging sign
    12: "traffic sign 3",  # high/big hanging sign
    13: "pole",
    14: "garbage-can",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground",
}

# mapped labels
learning_map = {
    0: 0,    # "unlabeled",
    1: 0,
    2: 0,
    3: 0,
    4: 1,    # "person",
    5: 1,    # "person",
    6: 2,    # "rider",
    7: 3,    # "car",
    8: 4,    # "trunk",
    9: 5,    # "plants",
    10: 6,   # "traffic sign"
    11: 6,   # "traffic sign"
    12: 6,   # "traffic sign"
    13: 7,   # "pole",
    14: 8,   # "garbage-can",
    15: 9,   # "building",
    16: 10,  # "cone/stone",
    17: 11,  # "fence",
    18: 0,
    19: 0,
    20: 0,
    21: 12,  # "bike",
    22: 13,  # "ground"
}


class SemanticPOSS(Dataset):

    N_LABELS = 20

    def __init__(self,
                 root,
                 split="training",
                 transform=None,
                 dataset_size=None,
                 multiframe_range=None,
                 skip_ratio=1,
                 skip_for_visu=1,
                 **kwargs):

        super().__init__(root, transform, None)

        self.split = split
        self.n_frames = 1
        self.multiframe_range = multiframe_range

        logging.info(f"SemanticPOSS - split {split}")

        # get the scenes
        assert(split in ["train", "val", "test", "verifying", "parametrizing"])
        if split == "verifying":
            raise NotImplementedError
        elif split == "parametrizing":
            raise NotImplementedError
        elif split == "train":
            # self.sequences = ['{:02d}'.format(i) for i in range(11) if i != 8]
            self.sequences = ['00', '01', '02', '04', '05']
        elif split == "val":
            # self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 8]
            self.sequences = ['03']
        elif split == "test":
            raise NotImplementedError
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', split)

        self.points_datapath = []
        self.labels_datapath = []
        for sequence in self.sequences:
            points_datapath = [path for path in Path(os.path.join(self.root, "dataset", "sequences", sequence, "velodyne")).rglob('*.bin')]
            points_datapath.sort()
            points_datapath = [path for path_id, path in enumerate(points_datapath) if (path_id % skip_ratio == 0)]
            self.points_datapath += points_datapath

        if skip_for_visu > 1:
            self.points_datapath = self.points_datapath[::skip_for_visu]

        for fname in self.points_datapath:
            fname = str(fname).replace("/velodyne/", "/labels/")
            fname = str(fname).replace(".bin", ".label")
            self.labels_datapath.append(fname)

        max_key = max(learning_map.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(learning_map.keys())] = list(learning_map.values())
        self.remap_lut_val = remap_lut_val

        # color map SK
        self.class_colors = np.array([
            [0, 0, 0],
            [245, 150, 100],
            [245, 230, 100],
            [150, 60, 30],
            [180, 30, 80],
            [255, 0, 0],
            [30, 30, 255],
            [200, 40, 255],
            [90, 30, 150],
            [255, 0, 255],
            [255, 150, 255],
            [75, 0, 75],
            [75, 0, 175],
            [0, 200, 255],
            [50, 120, 255],
            [0, 175, 0],
            [0, 60, 135],
            [80, 240, 150],
            [150, 240, 255],
            [0, 0, 255],
        ], dtype=np.uint8)
        self.class_colors = self.class_colors[[0,6,7,1,16,15,19,18,12,13,11,14,2,9]]

        logging.info(f"SemanticPOSS dataset {len(self.points_datapath)}")

    def get_weights(self):
        weights = torch.ones(self.N_LABELS)
        weights[0] = 0
        return weights

    @staticmethod
    def get_mask_filter_valid_labels(y):
        return (y > 0)

    def get_colors(self, labels):
        return self.class_colors[labels]

    def get_filename(self, index):
        fname = str(self.points_datapath[index])
        fname = str(fname).split("/")
        fname = ("_").join([fname[-3], fname[-1]])
        fname = fname[:-4]
        return fname

    @staticmethod
    def get_ignore_index():
        return 0

    @property
    def raw_file_names(self):
        return []

    def _download(self):  # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.points_datapath)

    def load_label_poss(self, label_path):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        sem_label = self.remap_lut_val[sem_label]
        return sem_label.astype(np.int32)

    def get(self, idx):
        """Get item."""

        fname_points = self.points_datapath[idx]
        frame_points = np.fromfile(fname_points, dtype=np.float32)

        pos = frame_points.reshape((-1, 4))
        intensities = pos[:, 3:]
        pos = pos[:, :3]

        # print(y)
        y = self.load_label_poss(self.labels_datapath[idx])
        unlabeled = y == 0

        # remove unlabeled points
        y = np.delete(y, unlabeled, axis=0)
        pos = np.delete(pos, unlabeled, axis=0)
        intensities = np.delete(intensities, unlabeled, axis=0)

        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0], 1), dtype=torch.float)

        return Data(x=x, intensities=intensities, pos=pos, y=y,
                    shape_id=idx,)
