from functools import partial
import numpy as np

import torch
import spconv.pytorch as spconv
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        LAYER_NUMS = model_cfg['LAYER_NUMS']
        LAYER_STRIDES = model_cfg['LAYER_STRIDES']
        NUM_FILTERS = model_cfg['NUM_FILTERS']
        UPSAMPLE_STRIDES = model_cfg['UPSAMPLE_STRIDES']
        NUM_UPSAMPLE_FILTERS = model_cfg['NUM_UPSAMPLE_FILTERS']

        if LAYER_NUMS is not None:
            assert len(LAYER_NUMS) == len(LAYER_STRIDES) == len(NUM_FILTERS)
            layer_nums = LAYER_NUMS
            layer_strides = LAYER_STRIDES
            num_filters = NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if UPSAMPLE_STRIDES is not None:
            assert len(UPSAMPLE_STRIDES) == len(NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = NUM_UPSAMPLE_FILTERS
            upsample_strides = UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, spatial_features):
        """
        Args:
            spatial_features
        Returns:
        """
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        return x


def post_act_block(
    in_channels,
    out_channels,
    kernel_size,
    indice_key=None,
    stride=1,
    padding=0,
    conv_type="subm",
    norm_fn=None,
):

    if conv_type == "subm":
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key
        )
    elif conv_type == "spconv":
        conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )
    elif conv_type == "inverseconv":
        conv = spconv.SparseInverseConv3d(
            in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False
        )
    elif conv_type == "transposeconv":
        conv = spconv.SparseConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key
        )
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None
    ):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels, 16, 3, padding=1, bias=False, indice_key="subm1"
            ),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key="subm1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(
                16,
                32,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv2",
                conv_type="spconv",
            ),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key="subm2"),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key="subm2"),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(
                32,
                64,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv3",
                conv_type="spconv",
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm3"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm3"),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(
                64,
                64,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=(0, 1, 1),
                indice_key="spconv4",
                conv_type="spconv",
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm4"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm4"),
        )

        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(
                64,
                128,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=last_pad,
                bias=False,
                indice_key="spconv_down2",
            ),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            "x_conv1": 16,
            "x_conv2": 32,
            "x_conv3": 64,
            "x_conv4": 64,
        }

    def forward(self, input_sp_tensor):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        return out


class HeightCompression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, encoded_spconv_tensor):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features


class AverageHeightCompression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, encoded_spconv_tensor):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.mean(2)
        return spatial_features


class BEVNet(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        self.bev_stride = 8
        point_cloud_range = np.array(kwargs['config'].DATASET.POINT_CLOUD_RANGE)
        voxel_size = kwargs['config'].DATASET.VOXEL_SIZE
        self.grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).astype(np.int64)[::-1] + [1, 0, 0]
        super().__init__()
        self.backbone_3d = VoxelBackBone8x(in_channels)
        model_cfg = {'LAYER_NUMS': [5, 5], 'LAYER_STRIDES': [1, 2], 'NUM_FILTERS': [128, 256], 'UPSAMPLE_STRIDES': [1, 2], 'NUM_UPSAMPLE_FILTERS': [256, 256]}
        self.backbone_2d = BaseBEVBackbone(model_cfg, out_channels)  # TODO fix
        self.height_compression = HeightCompression()
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(512, config.ENCODER.FEATURE_DIMENSION, kernel_size=1),
        #     nn.BatchNorm2d(config.ENCODER.FEATURE_DIMENSION, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv2d(config.ENCODER.FEATURE_DIMENSION, config.ENCODER.FEATURE_DIMENSION, kernel_size=1),
        #     nn.BatchNorm2d(config.ENCODER.FEATURE_DIMENSION, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv2d(config.ENCODER.FEATURE_DIMENSION, config.ENCODER.FEATURE_DIMENSION, kernel_size=1),
        #     nn.BatchNorm2d(config.ENCODER.FEATURE_DIMENSION, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv2d(config.ENCODER.FEATURE_DIMENSION, config.ENCODER.FEATURE_DIMENSION, kernel_size=1),
        # )
        self.final = nn.Conv2d(512, kwargs['config'].ENCODER.FEATURE_DIMENSION, kernel_size=1)
        # self.final = spconv.SparseConv3d(
        #     128, 128, 1, bias=False, indice_key="spconv_down2",
        # )

    def forward(self, voxels, coordinates):
        with torch.no_grad():
            bs = coordinates[-1, 0].item() + 1
        # voxels = torch.ones_like(voxels)
        sp_tensor = spconv.SparseConvTensor(
            features=voxels,
            indices=coordinates,
            spatial_shape=self.grid_size,
            batch_size=bs
        )
        sp_tensor = self.backbone_3d(sp_tensor)
        encoding = self.height_compression(sp_tensor)
        occupancy = encoding.any(1)

        sp_tensor = self.backbone_2d(encoding)
        sp_tensor = self.final(sp_tensor)
        return sp_tensor, occupancy
