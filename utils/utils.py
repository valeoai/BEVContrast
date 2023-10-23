import torch
import numpy as np


def confusion_matrix(preds, labels, num_classes):
    hist = (
        torch.bincount(
            num_classes * labels + preds,
            minlength=num_classes ** 2,
        )
        .reshape(num_classes, num_classes)
        .float()
    )
    return hist


def compute_IoU_from_cmatrix(hist, ignore_index=None):
    """Computes the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        m_IoU, fw_IoU, and matrix IoU
    """
    if ignore_index is not None:
        hist[ignore_index] = 0.0
    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
    IoU = intersection.float() / union.float()
    IoU[union == 0] = 1.0
    if ignore_index is not None:
        IoU = torch.cat((IoU[:ignore_index], IoU[ignore_index + 1:]))
    m_IoU = torch.mean(IoU).item()
    fw_IoU = (
        torch.sum(intersection) / (2 * torch.sum(hist) - torch.sum(intersection))
    ).item()
    return m_IoU, fw_IoU, IoU


def knn_classifier(train_features, train_labels, test_features, k, T, num_classes=1000, num_chunks=5000):
    preds = []
    train_features = train_features.t()
    num_test_images = test_features.shape[0]
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device, non_blocking=True)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx: min((idx + imgs_per_chunk), num_test_images), :
        ]
        batch_size = features.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        del similarity
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        # preds.append(torch.mode(retrieved_neighbors, dim=1)[0])

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        # distances_transform = distances.clone().div_(T).exp_()
        distances = distances.div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        preds.append(predictions[:, 0])

    return torch.cat(preds, 0)


def cosine_scheduler(base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
    # Code taken from https://github.com/facebookresearch/dino
    # Copyright (c) Facebook, Inc. and its affiliates.
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule


def transform_rotation(pc, seed):
    angle = seed * 2 * np.pi
    c = np.cos(angle)
    s = np.sin(angle)
    rotation = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    pc[:, :3] = pc[:, :3] @ rotation
    return pc


def transform_dilation(pc, seed):
    dilation = seed * 0.1 + 0.95
    pc[:, :3] = pc[:, :3] * dilation
    return pc


def transform_jittering(pc, seed):
    pc[:, 3] = np.random.normal(pc[:, 3], 0.01)
    return pc


def mask_points_outside_range(points, range):
    mask = (points[:, 0] >= range[0]) & (points[:, 0] <= range[3]) \
        & (points[:, 1] >= range[1]) & (points[:, 1] <= range[4])
    return points[mask]


def det_3x3(mat):

    a, b, c = mat[:, 0, 0], mat[:, 0, 1], mat[:, 0, 2]
    d, e, f = mat[:, 1, 0], mat[:, 1, 1], mat[:, 1, 2]
    g, h, i = mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2]

    det = a * e * i + b * f * g + c * d * h
    det = det - c * e * g - b * d * i - a * f * h

    return det


def det_2x2(mat):

    a, b = mat[:, 0, 0], mat[:, 0, 1]
    c, d = mat[:, 1, 0], mat[:, 1, 1]

    det = a * d - c * b

    return det


# def estimate_rot_trans(x, y, w):
#     # if threshold is not None:
#     #     w = w * (w > self.threshold).float()
#     w = torch.nn.functional.normalize(w, dim=-1, p=1)

#     # Center point clouds
#     mean_x = (w * x).sum(dim=-1, keepdim=True)
#     mean_y = (w * y).sum(dim=-1, keepdim=True)
#     x_centered = x - mean_x
#     y_centered = y - mean_y

#     # Covariance
#     cov = torch.bmm(y_centered, (w * x_centered).transpose(1, 2))

#     # Rotation
#     U, _, V = torch.svd(cov)
#     det = det_3x3(U) * det_3x3(V)
#     S = torch.eye(3, device=U.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
#     S[:, -1, -1] = det
#     R = torch.bmm(U, torch.bmm(S, V.transpose(1, 2)))

#     # Translation
#     T = mean_y - torch.bmm(R, mean_x)

#     return R, T, w


def estimate_rot_trans(x, y, w=None):
    if w is None:
        w = torch.ones(size=(x.shape[0], 1, x.shape[2]), device=x.device)
    # if threshold is not None:
    #     w = w * (w > self.threshold).float()
    w = torch.nn.functional.normalize(w, dim=-1, p=1)

    # Center point clouds
    mean_x = (w * x).sum(dim=-1, keepdim=True)
    mean_y = (w * y).sum(dim=-1, keepdim=True)
    x_centered = x - mean_x
    y_centered = y - mean_y

    # Covariance
    cov = torch.bmm(y_centered, (w * x_centered).transpose(1, 2))

    # Rotation
    U, _, V = torch.svd(cov)
    det = det_2x2(U) * det_2x2(V)
    S = torch.eye(2, device=U.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
    S[:, -1, -1] = det
    R = torch.bmm(U, torch.bmm(S, V.transpose(1, 2)))

    # Translation
    T = mean_y - torch.bmm(R, mean_x)

    return R, T


def compute_rte(t, t_est):

    t = t.squeeze().detach().cpu().numpy()
    t_est = t_est.squeeze().detach().cpu().numpy()

    return np.linalg.norm(t - t_est)


def compute_rre(R_est, R):

    eps = 1e-16

    R = R.squeeze().detach().cpu().numpy()
    R_est = R_est.squeeze().detach().cpu().numpy()

    return np.arccos(
        np.clip(
            np.trace(R_est.T @ R) / 2,
            # (np.trace(R_est.T @ R) - 1) / 2,
            -1 + eps,
            1 - eps
        )
    ) * 180. / np.pi
