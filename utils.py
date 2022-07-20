from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import cv2
import transforms3d.quaternions as txq
import transforms3d.euler as txe
from torchvision.datasets.folder import default_loader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import tools.camera_operator as cam_opt
from banet_track.ba_module import x_2d_coords_torch, batched_pi_inv, batched_transpose, batched_inv_pose, batched_pi


# from torch._six import container_abcs
from torch.utils.data._utils.collate import default_collate

import torch.nn as nn
import random

def random_shift(image, max_shift):
    """
    :param image: (B, 3, H, W)
    :param max_shift:( <= 4)
    :return: Randomly shift the input image via zero padding in x and y.
    """
    padX = random.randint(-max_shift, max_shift)
    padY = random.randint(-max_shift, max_shift)
    # padX:width, padY:height
    pad = nn.ZeroPad2d((padX, -padX, padY, -padY))

    return padX, padY, pad(image)


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)

    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    # keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    # keys_with_prefix['glr'] = 'glr'
    if 'obj_geod_th' in args_dict:
        keys_with_prefix['obj_geod_th'] = 'th'
    if 'loss_essential' in args_dict:
        keys_with_prefix['loss_essential'] = 'le'
    if 'loss_classif' in args_dict:
        keys_with_prefix['loss_classif'] = 'lc'
    keys_with_prefix['model'] = 'md'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value) or key == 'model':
            folder_string.append('{}{}'.format(prefix, value))
    # if args.pose_graph:
    #     folder_string.append('_pose_graph')
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path / timestamp


## NUMPY
def qlog(q):

    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        # assert q[0] <=1 and q[0] >= -1
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    :param poses_in: N x 12
    :param mean_t: 3
    :param std_t: 3
    :param align_R: 3 x 3
    :param align_t: 3
    :param align_s: 1
    :return: processed poses (translation + quaternion) N x 7   ?? actually, it's 6 [logq]
    :return: Camera pose matrix of (3x4):Tcw
    """
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    Tcws = np.zeros((len(poses_in), 3, 4))


    # align
    for i in range(len(poses_out)):
        pose_in = poses_in[i].reshape((3, 4))
        R = pose_in[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()
        Tcw = cam_opt.camera_pose_inv(R, pose_in[:3, 3])
        Tcws[i] = Tcw

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out, Tcws

def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N * 3 * 3
    :return: N * 4
    """
    rotdiff = R - R.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = R[:, 0, 0]
    t2 = R[:, 1, 1]
    t3 = R[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = Variable(torch.zeros(R.shape[0], 4)).float().cuda()
    q[:, 0] = torch.cos(theta / 2)
    q[:, 1:] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q

def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)

    except IOError as e:
        print(filename)
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None

    return img


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'viridis': high_res_colormap(cm.get_cmap('viridis')),
             'bone': cm.get_cmap('bone', 10000)}

def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    # uncertainty(single image) :poseudocolor or color gray image
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        array = tensor.squeeze().numpy()
        array = cv2.resize(array,None,fx=4,fy=4,interpolation=cv2.INTER_LINEAR)
        if max_value is None:
            max_value = np.max(array)
        #(H,W)
        norm_array = array / max_value
        # norm_array = cv2.resize(norm_array,None,fx=4,fy=4,interpolation=cv2.INTER_LINEAR)
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        #(3,H,W)
        x = array.transpose(2, 0, 1)[:3]

    # NCHW (color image)
    elif tensor.ndimension() == 4:
        if tensor.size(1) == 1:
            out_array=[]
            for i,stensor in enumerate(tensor):
                out = tensor2array(stensor,max_value,colormap)
                out_array.append(torch.from_numpy(out))
            x = vutils.make_grid(out_array,nrow=4,normalize=False,scale_each=True)
        else:
            assert (tensor.size(1) == 3)
            tensor = 0.5 + tensor * 0.5
            # array.astype('float32')
            x = vutils.make_grid(tensor, nrow=4, normalize=True, scale_each=True)
    #CHW
    elif tensor.ndimension() == 3:
        print("???")
        assert (tensor.size(0) == 3)
        tensor = 0.5 + tensor * 0.5
        # array.astype('float32')
        x = vutils.make_grid(tensor, nrow=4, normalize=True, scale_each=True)
    else:
        x = None
    return x


def save_checkpoint(save_path, pose_gen_state, is_best, num_epoch=0, filename='checkpoint_pth.pkl'):
    file_prefixes = ['pose_gen']
    states = [pose_gen_state]

    if is_best:
        for (prefix, state) in zip(file_prefixes, states):
            torch.save(state, save_path / '{}_{}_best_pth.pkl'.format(prefix, num_epoch))
        #for prefix in file_prefixes:
            #shutil.copyfile(save_path / '{}_{}_{}'.format(prefix, num_epoch, filename),
            #               save_path / '{}_{}_best_pth.pkl'.format(prefix, num_epoch))
    '''
    else:
        for (prefix, state) in zip(file_prefixes, states):
            torch.save(state, save_path / '{}_{}_{}'.format(prefix, num_epoch, filename))
    '''

def qexp(q):
  """
  Applies the exponential map to q
  :param q: (3,)
  :return: (4,)
  """
  n = np.linalg.norm(q)
  q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
  return q

def calc_vos_simple(poses):
  """
  calculate the VOs, from a list of consecutive poses
  :param poses: N x T x 7
  :return: N x (T-1) x 7
  """
  vos = []
  for p in poses:
    pvos = [p[i+1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p)-1)]
    vos.append(torch.cat(pvos, dim=0))
  vos = torch.stack(vos, dim=0)

  return vos

def show_batch(batch):
  npimg = batch.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
  plt.show()

def get_mask_2d(x_2d):
    # first coor of points  >= 0
    mask = torch.ge(x_2d[:, :, 0], 0)
    # print(x_2d.size())
    # print(mask.size())
    # assert mask.size()[0] == x_2d.size()[0]
    # assert mask.size()[1] == x_2d.size()[1]
    return mask

def preprocess_scene(x_2d, scene_depth, scene_K, scene_Tcw):
    # B*topK*C*H*W ? B=1?
    # N, L, C, H, W = scene_rgb.shape
    # scene_depth = scene_depth.view(N * L, 1, H, W)
    # scene_K = scene_K.view(N * L, 3, 3)
    # scene_Tcw = scene_Tcw.view(N * L, 3, 4)

    # generate 3D world position of scene
    d = scene_depth  # dim (N*L, H*W, 1)
    # 2D coordinates to camera 3D coordinates

    #depth should be m!  there seems to be cm!change in batched_pi_inv
    X_3d = batched_pi_inv(scene_K, x_2d, d)  # dim (N*L, H*W, 3)

    Rwc, twc = batched_inv_pose(R=scene_Tcw[:, :3, :3],
                                t=scene_Tcw[:, :3, 3].squeeze(-1))  # dim (N*L, 3, 3), （N, 3)
    # camera 3D to world 3D
    X_world = batched_transpose(Rwc.cuda(), twc.cuda(), X_3d)  # dim (N*L, H*W, 3)
    # X_world = X_world.view(N, L * H * W, 3)  # dim (N, L*H*W, 3)
    # scene_center = torch.mean(X_world, dim=1)  # dim (N, 3)
    # X_world -= scene_center.view(N, 1, 3)
    # X_world = batched_transpose(rand_R.cuda().expand(N, 3, 3),
    #                             torch.zeros(1, 3, 1).cuda().expand(N, 3, 1),
    #                             X_world)  # dim (N, L*H*W, 3), data augmentation
    # X_world = X_world.view(N, L, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()  # dim (N, L, 3, H, W)
    # scene_input = torch.cat((scene_rgb, X_world), dim=2)
    #
    # return scene_input.cuda(), scene_ori_rgb.cuda(), X_world.cuda(), \
    #        torch.gt(scene_depth, 1e-5).cuda().view(N, L, H, W), scene_center.cuda(), rand_R.expand(N, 3, 3).cuda()
    return X_world, torch.squeeze(torch.gt(scene_depth, 1e-5)) #(B, npoints, 3) (B, npoints)

def preprocess_single_scene(x_2d, scene_depth, scene_K, scene_Tcw):
    # generate 3D world position of scene
    d = scene_depth.unsqueeze(0)  # dim (1, npoints, 1)
    x_2d = x_2d.unsqueeze(0)
    scene_K = scene_K.unsqueeze(0)
    scene_Tcw = scene_Tcw.unsqueeze(0)
    # 2D coordinates to camera 3D coordinates

    #depth should be m!  there seems to be cm!change in batched_pi_inv
    X_3d = batched_pi_inv(scene_K, x_2d, d)  # dim (N*L, H*W, 3)

    Rwc, twc = batched_inv_pose(R=scene_Tcw[:, :3, :3],
                                t=scene_Tcw[:, :3, 3].squeeze(-1))  # dim (N*L, 3, 3), （N, 3)
    # camera 3D to world 3D
    X_world = batched_transpose(Rwc, twc, X_3d)  # dim (N*L, H*W, 3)
    return X_world.squeeze(0), torch.squeeze(torch.gt(scene_depth, 1e-5)) #(npoints, 3) (npoints)

def correspond_depth(x_2d, depth):
    """
    Geting corresponding depth of points in x_2d
    :param x_2d: npoints*2 (0-W, 1-H)
    :param depth: H*W
    :return: sparse_d :npoints*1 (0 for newly added (-1,-1)points)
    """
    npoints = x_2d.shape[0]
    sparse_d = torch.zeros(npoints)
    x_W = x_2d[:, 0].tolist()
    y_H = x_2d[:, 1].tolist()
    for j in range(npoints):
        h = int(y_H[j])
        w = int(x_W[j])
        if h >= 0 and w >= 0:
            sparse_d[j] = depth[h][w]
    return sparse_d.unsqueeze(1)

def diff_collation_fn(data_list):
    return torch.from_numpy(np.concatenate(data_list, 0))

# def custom_collation_fn(data_labels):
#     coords, feats, labels = list(zip(*data_labels))
#     # Generate batched coordinates
#     bcoords = ME.utils.batched_coordinates(coords)
#
#     # Concatenate all lists
#     feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
#
#     if isinstance(labels[0], container_abcs.Mapping):
#         labels_batch = {key: diff_collation_fn([d[key] for d in labels]) for key in labels[0]}
#
#     else:
#         print("label is not dict!")
#         labels_batch = torch.from_numpy(np.concatenate(labels, 0))
#
#     return bcoords, feats_batch, labels_batch
