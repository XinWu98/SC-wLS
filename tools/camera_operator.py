import numpy as np
import torch

def camera_pose_inv(R, t):
    """
    Compute the inverse pose
    """
    Rinv = R.transpose()
    Ov = - np.dot(Rinv, t)
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3, :3] = Rinv
    Tinv[:3, 3] = Ov
    return Tinv[:3, :]

def torch_camera_pose_inv(R, t):
    """
    Compute the inverse pose
    """
    Rinv = torch.transpose(R,1,0)
    Ov = - torch.mm(Rinv, t)
    Tinv = torch.empty([3,4])
    Tinv[:3, :3] = Rinv
    Tinv[:3, 3:4] = Ov
    return Tinv