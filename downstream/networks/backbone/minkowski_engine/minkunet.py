# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# https://arxiv.org/abs/2007.10985

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# https://arxiv.org/abs/2007.10985

import torch
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiNetwork
from MinkowskiEngine import MinkowskiReLU
import MinkowskiEngine.MinkowskiOps as me
import collections
from enum import Enum


class NormType(Enum):
    BATCH_NORM = 0
    SPARSE_LAYER_NORM = 1
    SPARSE_INSTANCE_NORM = 2
    SPARSE_SWITCH_NORM = 3


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.SPARSE_INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels, D=D)
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


class ConvType(Enum):
    """
    Define the kernel region type
    """

    HYPERCUBE = 0, "HYPERCUBE"
    SPATIAL_HYPERCUBE = 1, "SPATIAL_HYPERCUBE"
    SPATIO_TEMPORAL_HYPERCUBE = 2, "SPATIO_TEMPORAL_HYPERCUBE"
    HYPERCROSS = 3, "HYPERCROSS"
    SPATIAL_HYPERCROSS = 4, "SPATIAL_HYPERCROSS"
    SPATIO_TEMPORAL_HYPERCROSS = 5, "SPATIO_TEMPORAL_HYPERCROSS"
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = 6, "SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS "

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# Covert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
}

int_to_region_type = {i: m[0] for i, m in enumerate(ME.RegionType.__entries.values())}


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        # No temporal convolution
        if isinstance(kernel_size, collections.abc.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, collections.abc.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        # Define the CUBIC conv kernel for spatial dims and CROSS conv for temp dim
        axis_types = [
            ME.RegionType.HYPER_CUBE,
        ] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def conv(
    in_planes,
    out_planes,
    kernel_size,
    stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=-1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def conv_tr(
    in_planes,
    out_planes,
    kernel_size,
    upsample_stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=-1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        upsample_stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=upsample_stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def sum_pool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiSumPooling(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        kernel_generator=kernel_generator,
        dimension=D,
    )


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super(BasicBlockBase, self).__init__()

        self.conv1 = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            conv_type=conv_type,
            D=D,
        )
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            conv_type=conv_type,
            D=D,
        )
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM


class BottleneckBase(nn.Module):
    expansion = 4
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super(BottleneckBase, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            conv_type=conv_type,
            D=D,
        )
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
        self.norm3 = get_norm(
            self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum
        )

        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(BottleneckBase):
    NORM_TYPE = NormType.BATCH_NORM


class Model(MinkowskiNetwork):
    OUT_PIXEL_DIST = -1

    def __init__(self, in_channels, out_channels, config, D, **kwargs):
        super(Model, self).__init__(D)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config


class ResNetBase(Model):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    OUT_PIXEL_DIST = 32
    HAS_LAST_BLOCK = False
    CONV_TYPE = ConvType.HYPERCUBE

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        assert self.BLOCK is not None
        assert self.OUT_PIXEL_DIST > 0

        super(ResNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)

        self.network_initialization(in_channels, out_channels, config, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, config, D):
        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        dilations = config.dilations
        bn_momentum = config.opt.bn_momentum
        self.inplanes = self.INIT_DIM
        self.conv1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            D=D,
        )

        self.bn1 = get_norm(
            NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = sum_pool(
            kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D
        )

        self.layer1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[0], 1),
        )
        self.layer2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[1], 1),
        )
        self.layer3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[2], 1),
        )
        self.layer4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[3], 1),
        )

        self.final = conv(
            self.PLANES[3] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            D=D,
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_type=NormType.BATCH_NORM,
        bn_momentum=0.1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D,
                ),
                get_norm(
                    norm_type,
                    planes * block.expansion,
                    D=self.D,
                    bn_momentum=bn_momentum,
                ),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x


class Res16UNetBase(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    # CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE  # FOR ME0.5

    def __init__(self, in_channels, out_channels, config, D=3):
        try:
            self.projection = config["projection"]
        except KeyError:
            self.projection = False
        # self.normalize_feature = config["normalize_features"]
        super(Res16UNetBase, self).__init__(in_channels, out_channels, config, D)

    def network_initialization(self, in_channels, out_channels, config, D):
        dilations = self.DILATIONS
        bn_momentum = config["bn_momentum"]

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config["kernel_size"], 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )

        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr4p16s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr4 = get_norm(
            self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr5p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr5 = get_norm(
            self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr6p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr6 = get_norm(
            self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr7p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[7],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr7 = get_norm(
            self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        # if self.projection:
        #     self.bnproj = get_norm(
        #         self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum
        #     )
        #     self.proj = conv(
        #         self.PLANES[7], self.PLANES[7], kernel_size=1, stride=1, bias=True, D=D
        #     )

        # self.final = conv(
        #     self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D
        # )

        if out_channels > 0:
            self.final = conv(
                self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D
            )
        else:
            self.final = None

        self.relu = MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        encoder_out = self.block4(out)

        out = self.convtr4p16s2(encoder_out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = me.cat(out, out_p1)
        out = self.block8(out)

        # if self.projection:
        #     out = self.proj(out)
        #     out = self.bnproj(out)
        #     out = self.relu(out)

        if self.final is not None:
            out = self.final(out)

        # if self.normalize_feature:
        #     return SparseTensor(
        #         out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
        #         coordinate_map_key=out.coordinate_map_key,
        #         coordinate_manager=out.coordinate_manager,
        #     )
        # else:
        return out


class MinkEngMinkUNet(Res16UNetBase):
    BLOCK = BasicBlock

    def __init__(self, in_channels, out_channels, **kwargs):

        config = {"kernel_size": 3, "bn_momentum": 0.05}
        super().__init__(in_channels=in_channels, out_channels=out_channels, config=config)

    def forward(self, data, downstream=False):

        coords = torch.cat([data["voxel_coords_batch"].unsqueeze(1), data["voxel_coords"]], dim=1).int()
        feats = data["voxel_x"]
        input = ME.SparseTensor(feats, coords)

        outputs = super().forward(input).F

        vox_num = data["voxel_number"]
        increment = torch.cat([vox_num.new_zeros((1,)), vox_num[:-1]], dim=0)
        increment = increment.cumsum(0)
        increment = increment[data["batch"]]
        inv_map = data["voxel_to_pc_id"] + increment

        # interpolate the outputs
        outputs = outputs[inv_map]

        return outputs

    def get_last_layer_channels(self):
        return self.PLANES[-1]


class MinkUNet34(MinkEngMinkUNet):
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class MinkUNet18(MinkEngMinkUNet):
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
