import numpy as np
import torch



# ------------------------------------------------------
# Straight-through one-hot (Unchanged)
def straight_through_one_hot_from_probs(probs):
    idx = probs.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
    one_hot_st = (one_hot - probs).detach() + probs
    return one_hot_st, idx.squeeze(1)


def inv_log_signed(x):
    """Applies the inverse log-signed transformation (works with torch tensors)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def create_image_from_flat_tensor_torch(x_flat: torch.Tensor, path_valid_indices='/Users/sophieabramian/Documents/troposwhere/data/ml_data/valid_indices.npy', IMAGE_SIZE = 48):
    """
    Converts a batch of flat, D-dimensional tensors (VAE input format) 
    back into a 48x48 image tensor. Output: B x 1 x H x W.
    """
    valid_indices = np.load(path_valid_indices)
    valid_indices_torch = torch.from_numpy(valid_indices).long()

    B = x_flat.shape[0]
    device = x_flat.device
    out = torch.zeros(B, IMAGE_SIZE * IMAGE_SIZE, dtype=x_flat.dtype, device=device)
    out[:, valid_indices_torch.to(device)] = x_flat
    return out.view(B, 1, IMAGE_SIZE, IMAGE_SIZE)

def make_three_masks_torch(imgs: torch.Tensor, v1=-1.5, v2=1.5):
    """
    imgs: B x H x W or B x 1 x H x W
    returns masks: B x H x W (int64) with values {0,1,2}
    thresholds identical to your numpy version: < -1.5 ->0, -1.5..1.5->1, >1.5->2
    """
    if imgs.dim() == 4:
        imgs = imgs[:, 0]
    mask = torch.zeros_like(imgs, dtype=torch.int64)
    mask[imgs < v1] = 0
    mask[(imgs >= v1) & (imgs <= v2)] = 1
    mask[imgs > v2] = 2
    return mask