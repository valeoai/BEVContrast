import numpy as np


def revtrans_rotation(pc, trans_dict):
    angle = np.random.random() * 2 * np.pi
    c = np.cos(angle)
    s = np.sin(angle)
    rotation = np.array(
        [[c, -s], [s, c]], dtype=np.float32
    )
    pc[:, :2] = pc[:, :2] @ rotation
    rotation = np.pad(rotation, (0, 1))
    rotation[2, 2] = 1.
    trans_dict['R'] = rotation.T
    return pc, trans_dict


def revtrans_translation(pc, trans_dict):
    translation = np.clip(np.random.normal(size=2, scale=4.).astype(np.float32), -15, 15)  # no trans along z
    pc[:, :2] += translation
    trans_dict['T'] = np.pad(translation, (0, 1))
    return pc, trans_dict


def revtrans_jittering(pc, trans_dict):
    pc[:, 3] = np.random.normal(pc[:, 3], 0.01)
    return pc, trans_dict


def revtrans_scaling(pc, trans_dict):
    scale = np.random.uniform(0.95, 1.05)
    pc[:, :3] = pc[:, :3] * scale
    trans_dict['S'] = scale
    return pc, trans_dict
