import os
import numpy as np
import yaml
import logging
import argparse
import importlib

from tqdm import tqdm

from scipy.spatial import KDTree

# torch imports
import torch


from torch_geometric.data import DataLoader

from transforms import get_transforms, get_input_channels

import datasets
import networks
from networks.backbone import *


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser(description='Self supervised.')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--resultsDir', type=str, default="visus")
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--split', type=str, default="val")
    opts = parser.parse_args()

    logging.info("loading the config file")
    config = yaml.load(open(opts.config, "r"), yaml.FullLoader)

    logging.info("Dataset")
    DatasetClass = eval("datasets." + config["dataset_name"])
    test_transforms = get_transforms(config, train=False, downstream=True, keep_orignal_data=True)
    test_dataset = DatasetClass(config["dataset_root"],
                                split=opts.split,
                                transform=test_transforms,
                                )

    logging.info("Dataloader")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["threads"],
        follow_batch=["voxel_coords"]
    )

    num_classes = config["downstream"]["num_classes"]
    device = torch.device("cuda")

    logging.info("Network")
    if config["network"]["backbone_params"] is None:
        config["network"]["backbone_params"] = {}
    config["network"]["backbone_params"]["in_channels"] = get_input_channels(config["inputs"])
    config["network"]["backbone_params"]["out_channels"] = config["downstream"]["num_classes"]

    backbone_name = "networks.backbone."
    if config["network"]["framework"] is not None:
        backbone_name += config["network"]["framework"]
    importlib.import_module(backbone_name)
    backbone_name += "." + config["network"]["backbone"]
    net = eval(backbone_name)(**config["network"]["backbone_params"])
    net.to(device)
    net.eval()

    logging.info("Loading the weights from pretrained network")
    try:
        net.load_state_dict(torch.load(opts.ckpt), strict=True)
    except RuntimeError:
        ckpt = torch.load(opts.ckpt)
        ckpt = {k[4:]: v for k, v in ckpt['state_dict'].items()}
        net.load_state_dict(ckpt, strict=True)

    with torch.no_grad():
        t = tqdm(test_loader, ncols=100, disable=True)
        for data in t:

            data = data.to(device)

            # predictions
            predictions = net(data)
            predictions = torch.nn.functional.softmax(predictions[:, 1:], dim=1).max(dim=1)[1]
            predictions = predictions.cpu().numpy() + 1

            # interpolate to original point cloud
            original_pos_np = data["original_pos"].cpu().numpy()
            pos_np = data["pos"].cpu().numpy()
            tree = KDTree(pos_np)
            _, indices = tree.query(original_pos_np, k=1)
            predictions = predictions[indices]

            # update the confusion matric
            targets_np = data["original_y"].cpu().numpy()

            # create the colors
            prediction_colors = test_dataset.get_colors(predictions)
            target_colors = test_dataset.get_colors(targets_np)

            # good / bad predictions
            good_bad_pred = (predictions == targets_np).astype(np.uint8)

            fname = test_dataset.get_filename(data["shape_id"].item()) + ".xyz"

            # save everything
            predictions_dir = os.path.join(opts.resultsDir, "predictions")
            targets_dir = os.path.join(opts.resultsDir, "ground_truth")
            good_bad_pred_dir = os.path.join(opts.resultsDir, "good_bad")


            os.makedirs(predictions_dir, exist_ok=True)
            os.makedirs(targets_dir, exist_ok=True)
            os.makedirs(good_bad_pred_dir, exist_ok=True)


            np.savetxt(os.path.join(predictions_dir, fname),
                np.concatenate([original_pos_np, prediction_colors], axis=1),
                fmt=["%.3f", "%.3f", "%.3f", "%u", "%u", "%u"]
                )
            np.savetxt(os.path.join(targets_dir, fname),
                np.concatenate([original_pos_np, target_colors], axis=1),
                fmt=["%.3f", "%.3f", "%.3f", "%u", "%u", "%u"]
                )
            np.savetxt(os.path.join(good_bad_pred_dir, fname),
                np.concatenate([original_pos_np, good_bad_pred[:,np.newaxis]], axis=1),
                fmt=["%.3f", "%.3f", "%.3f", "%u"]
                )

            torch.cuda.empty_cache()

            print(fname, good_bad_pred.sum()/good_bad_pred.shape[0])
