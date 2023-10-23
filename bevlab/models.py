import torch
import numpy as np
import torch.nn as nn
from bevlab import backbones
import torch.nn.functional as F
from torch_scatter import scatter


class BEVTrainer(nn.Module):  # TODO rename
    def __init__(self, config):
        super().__init__()
        encoder_class = getattr(backbones, config.ENCODER.NAME)
        self.range = config.DATASET.POINT_CLOUD_RANGE
        self.voxel_size = config.DATASET.VOXEL_SIZE
        self.scale = self.range[3]
        self.encoder = encoder_class(config.ENCODER.IN_CHANNELS, config.ENCODER.OUT_CHANNELS, config=config)
        self.input_frames = config.DATASET.INPUT_FRAMES
        self.output_frames = config.DATASET.OUTPUT_FRAMES
        self.total_frames = self.input_frames + self.output_frames
        self.loss = config.OPTIMIZATION.LOSS
        self.bev_stride = config.OPTIMIZATION.BEV_STRIDE
        self.batch_first = config.ENCODER.COLLATE == "collate_minkowski"
        self.collapse = config.ENCODER.COLLATE != "collate_spconv"
        self.criterion = NCELoss(0.07)

    def collapse_to_bev(self, fmap, coordinates, pc_min, range, voxel_size, stride):
        # function handling the collapse of the feature map to the BEV plane
        coordinates[:, :2] = torch.div(coordinates[:, :2] + pc_min[:, :2] + range[3] / voxel_size, stride, rounding_mode='floor')
        coordinates = coordinates[:, [0, 1, 3]].long()
        res = int(range[3] / voxel_size / stride * 2)
        mask = torch.logical_and(torch.all(coordinates[:, :2] >= 0, 1), torch.all(coordinates[:, :2] < res, 1))
        coo = coordinates[mask]
        indices = coo[:, 2] * res * res + coo[:, 0] * res + coo[:, 1]
        bev = scatter(fmap[mask].permute(1, 0), indices, dim_size=(coordinates[-1, 2] + 1) * res * res, reduce='mean').reshape(-1, coordinates[-1, 2] + 1, res, res).permute(1,0,3,2)
        return bev

    def forward(self, batch):
        input_fmap = self.encoder(
            batch['voxels_in'], batch['coordinates_in'])
        if isinstance(input_fmap, tuple):
            input_fmap, occupancy_in = input_fmap
        device = input_fmap.device

        output_fmap = self.encoder(
            batch['voxels_out'], batch['coordinates_out'])
        if isinstance(output_fmap, tuple):
            output_fmap, occupancy_out = output_fmap

        if self.batch_first:
            batch['coordinates_in'] = batch['coordinates_in'][:, [1,2,3,0]]
            batch['coordinates_out'] = batch['coordinates_out'][:, [1,2,3,0]]
        if self.loss == "contrast_efficient":
            # alternative implementation of the loss, which is more memory efficient but slower
            input_bev = self.collapse_to_bev(input_fmap, batch['coordinates_in'], batch["pc_min_in"].cuda(), self.range, self.voxel_size, self.bev_stride)
            output_bev = self.collapse_to_bev(output_fmap, batch['coordinates_out'].clone(), batch["pc_min_out"].cuda(), self.range, self.voxel_size, self.bev_stride)
            occupancy_out = torch.any(output_bev != 0, 1)

            # recover R and T on the BEV plane
            R = (batch["R_out"].transpose(1, 2) @ batch["R_in"]).to(torch.float32)
            T = ((batch["T_in"] - batch["T_out"]).unsqueeze(1) @ batch["R_out"].to(torch.float32))
            Rim = R.transpose(1, 2)
            Tim = -T @ R
            Rim = Rim[:, :2, :2]
            Tim = Tim[:, 0, :2] / self.scale
            P = torch.cat([Rim, Tim.unsqueeze(2)], axis=2)
            grid = F.affine_grid(P.to(device, non_blocking=True), output_bev.shape, align_corners=False)

            pred_bev = torch.cat([F.grid_sample(input_bev[b:b+1], grid[b][occupancy_out[b]].view(1,-1,1,2), mode='bilinear', padding_mode='zeros', align_corners=False).squeeze().squeeze() for b in range(len(occupancy_out))], 1).T
            mask = pred_bev.sum(1) != 0
            s = mask.sum().item()
            output_bev = output_bev.permute(0, 2, 3, 1)[occupancy_out]
            if s < 4096:
                k = F.normalize(pred_bev[mask], p=2, dim=1)
                q = F.normalize(output_bev[mask], p=2, dim=1)
            else:
                c = np.random.choice(s, 4096, replace=False)
                k = F.normalize(pred_bev[mask][c], p=2, dim=1)
                q = F.normalize(output_bev[mask][c], p=2, dim=1)
            loss = self.criterion(k, q)
            return loss, dict()

        if self.collapse:
            input_bev = self.collapse_to_bev(input_fmap, batch['coordinates_in'], batch["pc_min_in"].cuda(), self.range, self.voxel_size, self.bev_stride)
            output_bev = self.collapse_to_bev(output_fmap, batch['coordinates_out'].clone(), batch["pc_min_out"].cuda(), self.range, self.voxel_size, self.bev_stride)
            occupancy_out = torch.any(output_bev != 0, 1)
        else:
            input_bev = input_fmap
            output_bev = output_fmap

        # recover R and T on the BEV plane
        R = (batch["R_out"].transpose(1, 2) @ batch["R_in"]).to(torch.float32)
        T = ((batch["T_in"] - batch["T_out"]).unsqueeze(1) @ batch["R_out"].to(torch.float32))
        Rim = R.transpose(1, 2)
        Tim = -T @ R
        Rim = Rim[:, :2, :2]
        Tim = Tim[:, 0, :2] / self.scale
        P = torch.cat([Rim, Tim.unsqueeze(2)], axis=2)
        grid = F.affine_grid(P.to(device, non_blocking=True), output_bev.shape, align_corners=False)

        pred_bev = F.grid_sample(input_bev[0::self.input_frames], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        if self.collapse:
            occupancy_pred = torch.any(output_bev != 0, 1)
        else:
            occupancy_pred = F.grid_sample(occupancy_in[0::self.input_frames].unsqueeze(1).to(torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(1).bool()
        mask = torch.logical_and(occupancy_out, occupancy_pred)

        if self.loss == 'contrast':
            # ===== PointContrast =====
            s = mask.sum().item()
            if s < 4096:
                k = F.normalize(pred_bev.permute(0, 2, 3, 1)[mask], p=2, dim=1)
                q = F.normalize(output_bev.permute(0, 2, 3, 1)[mask], p=2, dim=1)
            else:
                c = np.random.choice(s, 4096, replace=False)
                k = F.normalize(pred_bev.permute(0, 2, 3, 1)[mask][c], p=2, dim=1)
                q = F.normalize(output_bev.permute(0, 2, 3, 1)[mask][c], p=2, dim=1)
            loss = self.criterion(k, q)
            return loss, dict()
        else:
            raise Exception("Unknown loss")


def make_models(config):
    model = BEVTrainer(config)
    return model


class NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()

        loss = self.criterion(out, target)
        return loss