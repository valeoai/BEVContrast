

# me_found = importlib.util.find_spec("MinkowskiEngine") is not None
# logging.info(f"ME found - {torchsparse_found}")
# if me_found:
#     from MinkowskiEngine.utils import sparse_quantize as me_sparse_quantize
#     from MinkowskiEngine import SparseTensor as MESparseTensor


from MinkowskiEngine.utils import sparse_quantize as me_sparse_quantize
from MinkowskiEngine import SparseTensor as MESparseTensor
import torch
import math

def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])
    return torch.stack((rho, phi, input_xyz[:, 2]), dim=1)

class Quantize(object):

    def __init__(self, voxel_size, **kwargs):
        self.voxel_size = voxel_size
        self.cylinder_coords = kwargs["cylinder_coords"] if "cylinder_coords" in kwargs else False

    def __call__(self, data):

        if self.cylinder_coords:

            pc_ = cart2polar(data["pos"].clone())
            pc_[:,0] = pc_[:,0]/self.voxel_size # radius
            pc_[:,1] = pc_[:,1]/math.pi * 180 # angle (-180, 180)
            pc_[:,2] = pc_[:,2]/self.voxel_size # height
            pc_ = torch.round(pc_.clone())

        else:

            pc_ = torch.round(data["pos"].clone() / self.voxel_size)
        
        pc_ -= pc_.min(0, keepdim=True)[0]

        coords, indices, inverse_map = me_sparse_quantize(pc_,
                                            return_index=True,
                                            return_inverse=True)

        feats = data["x"][indices]

        data["voxel_coords"] = coords
        data["voxel_x"] = feats
        data["voxel_to_pc_id"] = inverse_map
        data["voxel_number"] = int(coords.shape[0])

        return data


# class MEQuantizeCylindrical(object):

#     def __init__(self, voxel_size) -> None:

        
#         self.voxel_size = voxel_size

#     def __call__(self, data):

#         pc_ = data["pos"].clone()
#         x, y, z = pc_[:,0], pc_[:,1], pc_[:,2]
#         rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
#         # corresponds to a split each 1°
#         phi = torch.atan2(y, x) * 180 / np.pi
#         z = z / self.voxel_size
#         pc_[:,0] = rho
#         pc_[:,1] = phi
#         pc_[:,2] = z

#         data["vox_pos"] = pc_

#         return data