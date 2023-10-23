import torch
import numpy as np
import cumm.tensorview as tv
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bevlab import datasets as datasets  # TODO
from torchsparse.utils.quantize import sparse_quantize as sparse_quantize_torchsparse
import MinkowskiEngine as ME
from torch.utils.data.distributed import DistributedSampler
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator


def collate_torchsparse(voxel_size, use_coords, list_data):
    batch = {}
    for key in list_data[0]:
        batch[key] = [l[key] for l in list_data]

    batch['voxels_in'], batch['voxels_out'] = [], []
    batch['indexes_in'], batch['indexes_out'] = [], []
    batch['inv_indexes_in'], batch['inv_indexes_out'] = [], []
    batch['batch_n_points_in'], batch['batch_n_points_out'] = [], []
    batch['pc_min_in'], batch['pc_min_out'] = [], []
    coords = []
    batch_id = 0
    offset = 0
    for group_pc in batch['points_in']:
        coords_aug = group_pc[:, :3] / voxel_size
        pc_ = np.round(coords_aug).astype(np.int32)
        pc_min = pc_.min(0, keepdims=1)
        pc_ -= pc_min

        coordinates, indexes, inv_indexes = sparse_quantize_torchsparse(pc_, return_index=True, return_inverse=True)
        coords.append(F.pad(torch.from_numpy(coordinates), (0, 1, 0, 0), value=batch_id))
        batch['batch_n_points_in'].append(coordinates.shape[0])
        pc_min = pc_min.repeat(indexes.shape[0], 0)
        batch['pc_min_in'].append(torch.from_numpy(pc_min))
        batch['indexes_in'].append(indexes)
        batch['inv_indexes_in'].append(inv_indexes + offset)
        if use_coords:
            batch['voxels_in'].append(torch.from_numpy(group_pc[indexes]))
        else:
            batch['voxels_in'].append(torch.from_numpy(group_pc[indexes, 3:]))
        batch_id += 1
        offset += coordinates.shape[0]
    batch['coordinates_in'] = torch.cat(coords, 0).int()
    batch['pc_min_in'] = torch.cat(batch['pc_min_in'])
    batch['voxels_in'] = torch.cat(batch['voxels_in'])

    coords = []
    batch_id = 0
    offset = 0
    for group_pc in batch['points_out']:
        coords_aug = group_pc[:, :3] / voxel_size
        pc_ = np.round(coords_aug).astype(np.int32)
        pc_min = pc_.min(0, keepdims=1)
        pc_ -= pc_min

        coordinates, indexes, inv_indexes = sparse_quantize_torchsparse(pc_, return_index=True, return_inverse=True)
        coords.append(F.pad(torch.from_numpy(coordinates), (0, 1, 0, 0), value=batch_id))
        batch['batch_n_points_out'].append(coordinates.shape[0])
        pc_min = pc_min.repeat(indexes.shape[0], 0)
        batch['pc_min_out'].append(torch.from_numpy(pc_min))
        batch['indexes_out'].append(indexes)
        batch['inv_indexes_out'].append(inv_indexes + offset)
        if use_coords:
            batch['voxels_out'].append(torch.from_numpy(group_pc[indexes]))
        else:
            batch['voxels_out'].append(torch.from_numpy(group_pc[indexes, 3:]))
        batch_id += 1
        offset += coordinates.shape[0]
    batch['coordinates_out'] = torch.cat(coords, 0).int()
    batch['pc_min_out'] = torch.cat(batch['pc_min_out'])
    batch['voxels_out'] = torch.cat(batch['voxels_out'])

    batch['R_in'] = torch.stack([torch.from_numpy(np.stack(R)) for R in batch['R_in']], axis=0)
    batch['R_out'] = torch.stack([torch.from_numpy(np.stack(R)) for R in batch['R_out']], axis=0)
    batch['T_in'] = torch.stack([torch.from_numpy(np.stack(T)) for T in batch['T_in']], axis=0)
    batch['T_out'] = torch.stack([torch.from_numpy(np.stack(T)) for T in batch['T_out']], axis=0)

    if 'flow' in batch:
        batch['flow'] = torch.from_numpy(np.stack(batch['flow']))

    return batch


def collate_minkowski(voxel_size, use_coords, list_data):
    batch = {}
    for key in list_data[0]:
        batch[key] = [l[key] for l in list_data]

    batch["voxels_in"], batch["voxels_out"] = [], []
    batch['indexes_in'], batch['indexes_out'] = [], []
    batch['inv_indexes_in'], batch['inv_indexes_out'] = [], []
    batch['batch_n_points_in'], batch['batch_n_points_out'] = [], []
    batch['pc_min_in'], batch['pc_min_out'] = [], []
    coords = []
    batch_id = 0
    for group_pc in batch["points_in"]:
        coords_aug = group_pc[:, :3] / voxel_size
        pc_ = np.round(coords_aug).astype(np.int32)
        pc_min = pc_.min(0, keepdims=1)
        pc_ -= pc_min

        coordinates, indexes, inv_indexes = ME.utils.quantization.sparse_quantize(pc_, return_index=True, return_inverse=True)
        coords.append(F.pad(coordinates, (1, 0, 0, 0), value=batch_id))
        batch['batch_n_points_in'].append(coordinates.shape[0])
        pc_min = pc_min.repeat(indexes.shape[0], 0)
        batch['pc_min_in'].append(torch.from_numpy(pc_min))
        batch["indexes_in"].append(indexes)
        batch['inv_indexes_in'].append(inv_indexes)
        if use_coords:
            batch["voxels_in"].append(torch.from_numpy(group_pc[indexes]))
        else:
            batch["voxels_in"].append(torch.from_numpy(group_pc[indexes, 3:]))
        batch_id += 1
    batch['coordinates_in'] = torch.cat(coords, 0).int()
    batch['pc_min_in'] = torch.cat(batch['pc_min_in'])
    batch['voxels_in'] = torch.cat(batch['voxels_in'])

    coords = []
    batch_id = 0
    for group_pc in batch["points_out"]:
        coords_aug = group_pc[:, :3] / voxel_size
        pc_ = np.round(coords_aug).astype(np.int32)
        pc_min = pc_.min(0, keepdims=1)
        pc_ -= pc_min

        coordinates, indexes, inv_indexes = ME.utils.quantization.sparse_quantize(pc_, return_index=True, return_inverse=True)
        coords.append(F.pad(coordinates, (1, 0, 0, 0), value=batch_id))
        batch['batch_n_points_out'].append(coordinates.shape[0])
        pc_min = pc_min.repeat(indexes.shape[0], 0)
        batch['pc_min_out'].append(torch.from_numpy(pc_min))
        batch["indexes_out"].append(indexes)
        batch['inv_indexes_out'].append(inv_indexes)
        if use_coords:
            batch["voxels_out"].append(torch.from_numpy(group_pc[indexes]))
        else:
            batch["voxels_out"].append(torch.from_numpy(group_pc[indexes, 3:]))
        batch_id += 1
    batch['coordinates_out'] = torch.cat(coords, 0).int()
    batch['pc_min_out'] = torch.cat(batch['pc_min_out'])
    batch['voxels_out'] = torch.cat(batch['voxels_out'])

    batch['R_in'] = torch.stack([torch.from_numpy(np.stack(R)) for R in batch['R_in']], axis=0)
    batch['R_out'] = torch.stack([torch.from_numpy(np.stack(R)) for R in batch['R_out']], axis=0)
    batch['T_in'] = torch.stack([torch.from_numpy(np.stack(T)) for T in batch['T_in']], axis=0)
    batch['T_out'] = torch.stack([torch.from_numpy(np.stack(T)) for T in batch['T_out']], axis=0)

    if 'flow' in batch:
        batch['flow'] = torch.from_numpy(np.stack(batch['flow']))

    return batch


class CollateSpconv:
    def __init__(self, config) -> None:

        self._voxel_generator = VoxelGenerator(
            vsize_xyz=config.DATASET.VOXEL_SIZE,
            coors_range_xyz=config.DATASET.POINT_CLOUD_RANGE,
            num_point_features=4,
            max_num_points_per_voxel=10,
            max_num_voxels=60000
        )
        self.coors_range = config.DATASET.POINT_CLOUD_RANGE

    def generate(self, points):
        voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

    @staticmethod
    def mask_points_by_range(points, limit_range):
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
            & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
        return points[mask]

    def collate_spconv(self, voxel_size, use_coords, list_data):
        batch = {}
        for key in list_data[0]:
            batch[key] = [l[key] for l in list_data]

        batch["voxels_in"], batch["voxels_out"] = [], []
        batch['indexes_in'], batch['indexes_out'] = [], []
        batch['inv_indexes_in'], batch['inv_indexes_out'] = [], []
        batch['batch_n_points_in'], batch['batch_n_points_out'] = [], []
        batch['pc_min_in'], batch['pc_min_out'] = [], []
        coords = []
        batch_id = 0
        for group_pc in batch["points_in"]:
            group_pc = self.mask_points_by_range(group_pc, self.coors_range)
            voxels, coordinates, num_points = self.generate(group_pc)
            coordinates = torch.from_numpy(coordinates)
            points_mean = torch.from_numpy(voxels).sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(torch.from_numpy(num_points).view(-1, 1), min=1.0).type_as(points_mean)
            points_mean = points_mean / normalizer
            voxels = points_mean.contiguous()

            coords.append(F.pad(coordinates, (1, 0, 0, 0), value=batch_id))
            pc_min = np.array([self.coors_range[0:3]]).repeat(coordinates.shape[0], 0)
            batch['pc_min_in'].append(torch.from_numpy(pc_min))
            if use_coords:
                batch["voxels_in"].append(voxels)
            else:
                batch["voxels_in"].append(voxels[:, 3:])
            batch_id += 1
        batch['coordinates_in'] = torch.cat(coords, 0).int()
        batch['pc_min_in'] = torch.cat(batch['pc_min_in'])
        batch['voxels_in'] = torch.cat(batch['voxels_in'])

        coords = []
        batch_id = 0
        for group_pc in batch["points_out"]:
            group_pc = self.mask_points_by_range(group_pc, self.coors_range)
            voxels, coordinates, num_points = self.generate(group_pc)
            coordinates = torch.from_numpy(coordinates)
            points_mean = torch.from_numpy(voxels).sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(torch.from_numpy(num_points).view(-1, 1), min=1.0).type_as(points_mean)
            points_mean = points_mean / normalizer
            voxels = points_mean.contiguous()

            coords.append(F.pad(coordinates, (1, 0, 0, 0), value=batch_id))
            pc_min = np.array([self.coors_range[0:3]]).repeat(coordinates.shape[0], 0)
            batch['pc_min_out'].append(torch.from_numpy(pc_min))
            if use_coords:
                batch["voxels_out"].append(voxels)
            else:
                batch["voxels_out"].append(voxels[:, 3:])
            batch_id += 1
        batch['coordinates_out'] = torch.cat(coords, 0).int()
        batch['pc_min_out'] = torch.cat(batch['pc_min_out'])
        batch['voxels_out'] = torch.cat(batch['voxels_out'])

        batch['R_in'] = torch.stack([torch.from_numpy(np.stack(R)) for R in batch['R_in']], axis=0)
        batch['R_out'] = torch.stack([torch.from_numpy(np.stack(R)) for R in batch['R_out']], axis=0)
        batch['T_in'] = torch.stack([torch.from_numpy(np.stack(T)) for T in batch['T_in']], axis=0)
        batch['T_out'] = torch.stack([torch.from_numpy(np.stack(T)) for T in batch['T_out']], axis=0)

        if 'flow' in batch:
            batch['flow'] = torch.from_numpy(np.stack(batch['flow']))

        return batch


collate_fns = {"collate_torchsparse": collate_torchsparse,
               "collate_minkowski": collate_minkowski}


def make_dataloader(config, phase, world_size=1, rank=0):
    dataset_class = getattr(datasets, config.DATASET.TRAIN)
    dataset = dataset_class(phase, config)
    try:
        collate_fn = collate_fns[config.ENCODER.COLLATE]
    except KeyError:
        collate_fn = CollateSpconv(config).collate_spconv

    use_coords = config.ENCODER.IN_CHANNELS == 4
    collate_fn = partial(collate_fn, config.DATASET.VOXEL_SIZE, use_coords)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        return DataLoader(
            dataset,
            batch_size=config.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            shuffle=False,
            num_workers=config.OPTIMIZATION.NUM_WORKERS_PER_GPU,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=sampler,
            persistent_workers=True
        )
    return DataLoader(
        dataset,
        batch_size=config.OPTIMIZATION.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=config.OPTIMIZATION.NUM_WORKERS_PER_GPU,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda id: np.random.seed(
            torch.initial_seed() // 2 ** 32 + id
        ),
        persistent_workers=True
    )
