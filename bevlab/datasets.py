import os
import numpy as np
import os.path as osp
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from bevlab.transforms import revtrans_rotation, revtrans_translation, revtrans_scaling
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


class NuscenesDataset(Dataset):
    CUSTOM_SPLIT = [
        "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
        "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
        "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
        "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
        "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
        "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
        "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
        "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
        "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
        "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
        "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
        "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
        "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
        "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
        "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
        "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
        "scene-1058", "scene-1094", "scene-1098", "scene-1107",
    ]

    def __init__(
        self,
        phase,
        config,
        **kwargs,
    ):
        self.phase = phase
        self.dataset_root = config.DATASET.DATASET_ROOT
        self.data_root = osp.join(self.dataset_root, 'data')
        self.num_frames_in = config.DATASET.INPUT_FRAMES
        self.num_frames_out = config.DATASET.OUTPUT_FRAMES
        self.num_frames = self.num_frames_in + self.num_frames_out
        self.select = config.DATASET.SKIP_FRAMES + 1
        self.voxel_size = config.DATASET.VOXEL_SIZE
        self.bev_stride = config.OPTIMIZATION.BEV_STRIDE
        self.apply_scaling = config.DATASET.APPLY_SCALING

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        elif config.DEBUG:
            self.nusc = NuScenes(
                version="v1.0-mini", dataroot=self.dataset_root, verbose=False
            )
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot=self.dataset_root, verbose=False
            )

        self.frame_list = list()
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(self.CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = self.CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                current_sample_token = scene["first_sample_token"]
                # Loop to get all successive keyframes
                sequence = []
                while current_sample_token != "":
                    current_sample = self.nusc.get("sample", current_sample_token)
                    sequence.append(current_sample)
                    current_sample_token = current_sample["next"]

                # Add new scans in the list
                for i in range(len(sequence) - self.num_frames * self.select + 1):
                    self.frame_list.append([sequence[j] for j in range(i, i + self.num_frames * self.select, self.select)])

    def load_point_cloud(self, sample):
        pointsensor = self.nusc.get("sample_data", sample["LIDAR_TOP"])
        pcl_path = osp.join(self.nusc.dataroot, pointsensor["filename"])
        points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        points[:, 3] = points[:, 3] / 255
        cs_record = self.nusc.get(
            "calibrated_sensor", pointsensor["calibrated_sensor_token"]
        )
        Re = Quaternion(cs_record['rotation']).rotation_matrix.astype(np.float32)
        Te = np.array(cs_record["translation"], dtype=np.float32)
        poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
        Rw = Quaternion(poserecord['rotation']).rotation_matrix.astype(np.float32)
        Tw = np.array(poserecord["translation"], dtype=np.float32)
        R = Rw @ Re
        T = Rw @ Te + Tw
        return points, R, T

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        return_dict = dict()

        pc, R, T = self.load_point_cloud(self.frame_list[idx][0]['data'])

        trans_dict = {}
        pc, trans_dict = revtrans_translation(pc, trans_dict)
        pc, trans_dict = revtrans_rotation(pc, trans_dict)
        T -= R @ trans_dict['T']
        R = R @ trans_dict['R'].T
        if self.apply_scaling:
            pc, trans_dict = revtrans_scaling(pc, trans_dict)
            R = R * trans_dict['S']
        anns = []
        for ann in self.frame_list[idx][0]['anns']:
            ann = self.nusc.get('sample_annotation', ann)
            if ann['num_lidar_pts'] == 0:
                continue
            anns.append([np.array(ann['translation']), ann['instance_token']])

        return_dict["points_in"] = pc
        return_dict["R_in"] = R
        return_dict["T_in"] = T

        pc, R, T = self.load_point_cloud(self.frame_list[idx][1]['data'])

        trans_dict = {}
        pc, trans_dict = revtrans_translation(pc, trans_dict)
        pc, trans_dict = revtrans_rotation(pc, trans_dict)
        T -= R @ trans_dict['T']
        R = R @ trans_dict['R'].T

        return_dict["points_out"] = pc
        return_dict["R_out"] = R
        return_dict["T_out"] = T

        return return_dict


class SemanticKITTIDataset(Dataset):
    TRAIN_SET = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}
    VALIDATION_SET = {8}
    TEST_SET = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

    def __init__(
        self,
        phase,
        config,
        **kwargs,
    ):

        if phase in ("val", "validation", "verifying"):
            phase_set = self.VALIDATION_SET
        else:
            phase_set = self.TRAIN_SET
        self.dataset_root = config.DATASET.DATASET_ROOT
        self.data_root = osp.join(self.dataset_root, 'dataset/sequences')
        self.num_frames_in = config.DATASET.INPUT_FRAMES
        self.num_frames_out = config.DATASET.OUTPUT_FRAMES
        self.num_frames = self.num_frames_in + self.num_frames_out
        self.select = config.DATASET.SKIP_FRAMES + 1
        self.voxel_size = config.DATASET.VOXEL_SIZE
        self.bev_stride = config.OPTIMIZATION.BEV_STRIDE
        self.apply_scaling = config.DATASET.APPLY_SCALING

        self.frame_list = list()
        self.calib_tr = []
        for seq_n, num in enumerate(phase_set):
            directory = next(
                os.walk(
                    f"{self.data_root}/{num:0>2d}/velodyne"
                )
            )
            directory_sorted = np.sort(directory[2])
            poses = np.loadtxt(f"{self.data_root}/{num:0>2d}/poses.txt", dtype=np.float32).reshape(-1, 3, 4)
            poses = np.pad(poses, ((0,0),(0,1),(0,0)))
            poses[:, 3, 3] = 1.
            with open(f"{self.data_root}/{num:0>2d}/calib.txt", "r") as calib_file:
                line = calib_file.readlines()[-1]
                assert line.startswith("Tr"), f"There is an issue with calib.txt in scene {num}"
                content = line.strip().split(":")[1]
                values = [float(v) for v in content.strip().split()]
                pose = np.zeros((4, 4), dtype=np.float32)
                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0
                pose_inv = np.linalg.inv(pose)
                self.calib_tr.append((pose, pose_inv))
            sequence = list(
                map(
                    lambda x: f"{self.data_root}/"
                    f"{num:0>2d}/velodyne/" + x,
                    directory_sorted,
                )
            )
            for i in range(len(sequence) - self.num_frames * self.select + 1):
                self.frame_list.append([(sequence[j], poses[j], seq_n) for j in range(i, i + self.num_frames * self.select, self.select)])
            seq_n += 1

    def load_point_cloud(self, frame, calib):
        points = np.fromfile(frame[0], dtype=np.float32).reshape((-1, 4))
        pose = frame[1]
        pose = calib[1] @ (pose @ calib[0])
        R = pose[:, :3]
        T = pose[:, 3]
        return points, R, T

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        return_dict = dict()

        pc, R, T = self.load_point_cloud(self.frame_list[idx][0], self.calib_tr[self.frame_list[idx][0][2]])

        trans_dict = {}
        pc, trans_dict = revtrans_translation(pc, trans_dict)
        pc, trans_dict = revtrans_rotation(pc, trans_dict)
        T -= R @ trans_dict['T']
        R = R @ trans_dict['R'].T
        if self.apply_scaling:
            pc, trans_dict = revtrans_scaling(pc, trans_dict)
            R = R * trans_dict['S']

        return_dict["points_in"] = pc
        return_dict["R_in"] = R
        return_dict["T_in"] = T

        pc, R, T = self.load_point_cloud(self.frame_list[idx][1],
                                         self.calib_tr[self.frame_list[idx][1][2]])

        trans_dict = {}
        pc, trans_dict = revtrans_translation(pc, trans_dict)
        pc, trans_dict = revtrans_rotation(pc, trans_dict)
        T -= R @ trans_dict['T']
        R = R @ trans_dict['R'].T

        return_dict["points_out"] = pc
        return_dict["R_out"] = R
        return_dict["T_out"] = T

        return return_dict
