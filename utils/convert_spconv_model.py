import sys
import torch


if __name__ == "__main__":
    path = sys.argv[1]
    ckpt = torch.load(path)
    ckpt = ckpt['state_dict']
    ckpt = {k.replace('encoder.', ''): v for k, v in ckpt.items()}
    del ckpt['final.weight']
    torch.save({'model_state': ckpt}, path.replace('.pt', '_converted.pt'))
