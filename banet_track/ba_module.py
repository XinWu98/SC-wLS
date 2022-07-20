import torch
import numpy as np


def x_2d_coords_torch(n, h, w):
    # 0-x-W, 1-y-H
    x_2d = np.zeros((n, h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[:, y, :, 1] = y
    for x in range(0, w):
        x_2d[:, :, x, 0] = x
    return torch.Tensor(x_2d)


""" Camera Operations --------------------------------------------------------------------------------------------------
"""


def batched_inv_pose(R, t):
    """
    Compute the inverse pose [Verified]
    :param R: rotation matrix, dim (N, 3, 3)
    :param t: translation vector, dim (N, 3)
    :return: inverse pose of [R, t]
    """
    N = R.size(0)
    Rwc = torch.transpose(R, 1, 2)
    tw = -torch.bmm(Rwc, t.view(N, 3, 1))
    return Rwc, tw


def batched_pi(K, X):
    """
    Projecting the X in camera coordinates to the image plane
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :param X: point position in 3D camera coordinates system, is a 3D array with dimension of (N, num_points, 3)
    :return: N projected 2D pixel position u (N, num_points, 2) and the depth X (N, num_points, 1)
    """
    # cx = W/2
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    X[:,:,2].clamp_(1e-5)  # avoid division by zero
    #!!!! /X[:,:,2:3]: Not be zero!!!
    u_x = fx * X[:, :, 0:1] / X[:, :, 2:3] + cx
    u_y = fy * X[:, :, 1:2] / X[:, :, 2:3] + cy
    u = torch.cat([u_x, u_y], dim=-1)
    return u, X[:, :, 2:3]


def batched_correspond_depth(x_2d, depth):
    """
    Geting corresponding depth of points in x_2d
    :param x_2d: B*npoints*2 (0-W, 1-H)
    :param depth: B*H*W
    :return: sparse_d : B*npoints*1 (0 for newly added (-1,-1)points)
    """
    B = x_2d.shape[0]
    npoints = x_2d.shape[1]
    sparse_d = torch.zeros(B, npoints)
    x_W = x_2d[:, :, 0].tolist()
    y_H = x_2d[:, :, 1].tolist()
    for i in range(B):
        for j in range(npoints):
            h = int(y_H[i][j])
            w = int(x_W[i][j])
            if h >= 0 and w >= 0:
                sparse_d[i][j] = depth[i][h][w]
                # print(depth[i][h][w])
            # else:
            #     sparse_d[i][j][0] = 1e-8

    return sparse_d.unsqueeze(2).cuda()


def batched_pi_inv(K, x, d):
    """
    Projecting the pixel in 2D image plane and the depth to the 3D point in camera coordinate.
    :param x: 2d pixel position, a 2D array with dimension of (N, num_points, 2)
    :param d: depth at that pixel, a array with dimension of (N, num_points, 1)
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :return: 3D point in camera coordinate (N, num_points, 3)
    """
    # print(K.size())
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    # print(d.device,fx.device, x.device, cx.device)
    # print(d.dtype, fx.dtype, cx.dtype,x[0,0,0].dtype)
    # mm!!
    # d = d * 0.001
    # using meters! (attention when load depth!!!)
    X_x = d * (x[:, :, 0:1] - cx) / fx
    X_y = d * (x[:, :, 1:2] - cy) / fy
    X_z = d
    X = torch.cat([X_x, X_y, X_z], dim=-1)
    return X


def batched_transpose(R, t, X):
    """
    Pytorch batch version of computing transform of the 3D points
    :param R: rotation matrix in dimension of (B, 3, 3)
    :param t: translation vector (B, 3)
    :param X: points with 3D position, a 2D array with dimension of (B, num_points, 3). (0,0,0)for added data
    :return: transformed 3D points
    """
    assert R.shape[1] == 3
    assert R.shape[2] == 3
    assert t.shape[1] == 3
    N = R.shape[0]
    M = X.shape[1]
    # print(R.device, X.device)
    # print(R.dtype, X.dtype)
    X_after_R = torch.matmul(R, torch.transpose(X, 1, 2))
    X_after_R = torch.transpose(X_after_R, 1, 2)
    # print(X_after_R.size(), t.size())
    trans_X = X_after_R + t.view(N, 1, 3).expand(N, M, 3)
    return trans_X
