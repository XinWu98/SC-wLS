import os.path as osp
from torchvision import transforms
import torch
import numpy as np
from numpy import mat
from torch.utils import data
import math
import cv2
import random
import torch.nn.functional as F
from skimage.transform import rotate, resize
from skimage import io
from utils import qlog
from tools.pose_transformations import quaternion_from_matrix, euler_from_quaternion, euler_from_matrix, \
    quaternion_from_euler, quaternion_matrix
import tools.camera_operator as cam_opt
import time

OUTPUT_SUBSAMPLE = 8

def rot2euler(p):
    """
    convert 6DoF pose with rotation matrix to 6DoF euler angle
    :param p: vector--12
    :return: vector--6
    """
    R = mat([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
    euler = euler_from_matrix(matrix=R)
    return np.array([p[3], p[7], p[11], euler[0], euler[1], euler[2]], np.float)


def rot2quat(p):
    """"
    convert 6DoF pose with rotation matrix to 6DoF quaternion
    : param p: vector -- 12
    : return vector -- 7
    """
    R = mat([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
    quat = quaternion_from_matrix(matrix=R)
    return np.array([p[3], p[7], p[11], quat[0], quat[1], quat[2], quat[3]])


def rot2logq(p):
    """
    convert 6DoF pose with rotation matrix to 6DoF logq
    :param p: vector -- 12
    :return: vector -- 6
    """
    R = mat([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
    q = np.array(quaternion_from_matrix(matrix=R))
    q *= np.sign(q[0])
    logq = qlog(q)
    return np.array([p[3], p[7], p[11], logq[0], logq[1], logq[2]], np.float)


def q2logq(p):
    q = p[3:]
    q *= np.sign(q[0])
    logq = qlog(q)
    return np.array([p[0], p[1], p[2], logq[0], logq[1], logq[2]], np.float)


def q2T(p):
    cam_rot = [float(r) for r in p[3:]]

    # quaternion to axis-angle
    angle = 2 * math.acos(cam_rot[0])
    x = cam_rot[1] / math.sqrt(1 - cam_rot[0] ** 2)
    y = cam_rot[2] / math.sqrt(1 - cam_rot[0] ** 2)
    z = cam_rot[3] / math.sqrt(1 - cam_rot[0] ** 2)

    cam_rot = [x * angle, y * angle, z * angle]

    cam_rot = np.asarray(cam_rot)
    cam_rot, _ = cv2.Rodrigues(cam_rot)

    cam_trans = [float(r) for r in p[0:3]]
    cam_trans = np.asarray([cam_trans])
    cam_trans = np.transpose(cam_trans)
    cam_trans = - np.matmul(cam_rot, cam_trans)

    if np.absolute(cam_trans).max() > 10000:
        print("Skipping image, Extremely large translation. Outlier?")
        print(cam_trans)
        return None
    cam_pose = np.concatenate((cam_rot, cam_trans), axis=1)
    cam_pose = np.concatenate((cam_pose, [[0, 0, 0, 1]]), axis=0)
    return cam_pose

def q2T(p):
    cam_rot_T = quaternion_matrix(p[3:])
    cam_trans = np.array(p[:3], dtype=np.float64, copy=True)
    if np.absolute(cam_trans).max() > 10000:
        print("Skipping image, Extremely large translation. Outlier?")
        print(cam_trans)
        return None
    cam_rot_T[0:3,3] = cam_trans
    return cam_rot_T

def q2euler(p):
    euler = euler_from_quaternion(p[3:])
    return np.array([p[0], p[1], p[2], euler[0], euler[1], euler[2]], np.float)

class Cambridge(data.Dataset):
    def __init__(self, scene, data_path, train=True, transform=None,
                 target_transform=None, seed=7, steps=1,
                 keep_first=True,
                 skip=1, variable_skip=False,
                 preload_image=True,
                 rot_model='T',
                 augment=False,
                 aug_rotation=30,
                 aug_scale_min=2 / 3,
                 aug_scale_max=5 / 4,
                 aug_contrast=0.1,
                 aug_brightness=0.1,
                 image_height=480,
                 ):
        self.variable_skip = variable_skip
        self.transform = transform
        self.target_tranform = target_transform
        self.steps = steps
        self.skip = skip
        self.keep_first = keep_first
        self.train = train
        self.rot_model = rot_model
        self.preload_image = preload_image

        np.random.seed(seed)
        self.image_height = image_height
        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_contrast = aug_contrast
        self.aug_brightness = aug_brightness
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_height),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4],  
                std=[0.25]
            )
        ])
        self.sparse = True
        if self.steps == 1:
            self.skip = 0
        begin = time.time()

        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)
        # data_dir = osp.join('data', 'temporal', 'Cambridge', scene)
        # if not osp.isdir(data_dir):
        #     data_dir = osp.join('temporal', 'Cambridge', scene)

        if train:
            image_list = osp.join(base_dir, 'dataset_train.txt')
        else:
            image_list = osp.join(base_dir, 'dataset_test.txt')

        f = open(image_list)
        self.camera_list = f.readlines()
        f.close()
        self.camera_list = self.camera_list[3:]
        self.camera_list.sort()

        # read poses and collect image names
        self.c_imgs = []
        self.sc_imgs = []
        self.calibration_files=[]
        self.poses = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        self.samples = []
        sequence_length = self.skip * (self.steps - 1)
        seq_num = []
        seq_name = []

        def get_indices(index):
            if self.variable_skip:
                skips = np.random.randint(1, high=self.skip, size=self.steps - 1)
            else:
                skips = self.skip * np.ones(self.steps - 1)

            offsets = np.insert(skips, 0, 0).cumsum()
            offsets = offsets.astype(np.int)[::-1]
            idx = index - offsets

            assert np.all(idx >= 0), '{:d}'.format(index)
            return idx

        last_name = ''
        now_num = 0
        valid_num = 0
        for idx, camera in enumerate(self.camera_list):
            elements = camera.split()
            image_name = elements[0]
            pose_qt = elements[1:]
            
            pose_inv = q2T(pose_qt)
            if pose_inv is None:
                print(image_name)
                continue
            pose = np.linalg.inv(pose_inv)

            self.c_imgs.append(osp.join(base_dir, image_name))

            seq, num = image_name.split('/')
            
            if train:
                self.sc_imgs.append(osp.join(data_path, 'cambridge_init', '{}_train'.format(scene),
                                             '{:s}_{:s}'.format(seq, num.replace('png', 'dat'))))
            self.poses.append(pose)

            # Do not use idx,since there is invalid pose and be excluded.Use valid_num.
            if valid_num == 0:
                last_name = seq
                now_num = 0
                seq_name.append(seq)
                if valid_num == sequence_length:
                    self.samples.append(get_indices(valid_num))
            else:
                if last_name == seq:
                    now_num += 1
                    if now_num >= sequence_length:
                        self.samples.append(get_indices(valid_num))
                else:
                    last_name = seq
                    seq_name.append(seq)
                    seq_num.append(now_num + 1)
                    now_num = 0
                    if now_num >= sequence_length:
                        self.samples.append(get_indices(valid_num))
            valid_num += 1
        seq_num.append(now_num + 1)

        if self.preload_image:
            self.loaded_cimgs = [io.imread(fn) for fn in self.c_imgs]
            if train:
                self.loaded_scs = [torch.load(fn) for fn in self.sc_imgs]

        self.img_k = np.asarray([[744.375, 0, 426], [0, 744.375, 240], [0, 0, 1]], dtype=np.float32)
        self.pose_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print("ReadData time:", time.time() - begin)

    def process_pose(self, pose_in):
        if self.rot_model == 'quat':
            pose_out = pose_in
        elif self.rot_model == 'logq':
            pose_out = q2logq(pose_in)
        elif self.rot_model == 'euler':
            pose_out = q2euler(pose_in)
        elif self.rot_model == 'T':
            pose_out = q2T(pose_in)
        else:
            raise "No such rotation model %s" % self.rot_model
        return pose_out

    def download(self):
        pass

    def process(self):
        pass

    def convert_pose(self, p):
        if self.rot_model == 'rot':
            poses_rel = p
        elif self.rot_model == 'quat':
            poses_rel = rot2quat(p)
        elif self.rot_model == 'euler':
            poses_rel = rot2euler(p)
        elif self.rot_model == 'logq':
            poses_rel = rot2logq(p)
        else:
            raise "No such rot mode %s" % self.rot_model

        return poses_rel

    def my_rot(self, t, angle, order, mode='constant'):
        t = t.permute(1, 2, 0).numpy()
        t = rotate(t, angle, order=order, mode=mode)
        t = torch.from_numpy(t).permute(2, 0, 1).float()
        return t

    def __getitem__(self, index):
        indexs = self.samples[index]
        frame = dict()
        poses = torch.stack([torch.from_numpy(self.poses[i]).float() for i in indexs], dim=0)
        K_tensor = torch.stack([torch.from_numpy(self.img_k.copy()) for i in indexs], dim=0)

        if self.preload_image:
            images = [self.loaded_cimgs[i] for i in indexs]
        else:
            images = [io.imread(self.c_imgs[i]) for i in indexs]
        if self.augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            # augment input image
            cur_image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(self.image_height * scale_factor)),
                transforms.ColorJitter(brightness=self.aug_brightness, contrast=self.aug_contrast),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],
                    std=[0.25]
                )
            ])

            images = torch.stack(
                    [self.my_rot(cur_image_transform(image), angle, 1, 'reflect') for image in images], dim=0)

            if self.train:
                if self.sparse:
                    # rotate and scale initalization targets
                    coords_w = math.ceil(images.size(3) / OUTPUT_SUBSAMPLE)
                    coords_h = math.ceil(images.size(2) / OUTPUT_SUBSAMPLE)
                    coords_set = []
                    for i in indexs:
                        if self.preload_image:
                            coords = self.loaded_scs[i]
                        else:
                            coords = torch.load(self.sc_imgs[i])
                            # (3,H,W)
                        coords = F.interpolate(coords.unsqueeze(0), size=(coords_h, coords_w))[0]
                        coords = self.my_rot(coords, angle, 0).view(3, -1).permute(1, 0)
                        coords_set.append(coords)
                    frame['coords'] = torch.stack(coords_set, dim=0)

            angle = angle * math.pi / 180
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle)
            pose_rot[0, 1] = -math.sin(angle)
            pose_rot[1, 0] = math.sin(angle)
            pose_rot[1, 1] = math.cos(angle)
            # scale focal length
            K_tensor[:, 0, 0] *= scale_factor
            K_tensor[:, 1, 1] *= scale_factor
            # image center
            K_tensor[:, 0, 2] = images.size(3) / 2
            K_tensor[:, 1, 2] = images.size(2) / 2
            # Broadcast
            poses = torch.matmul(poses, pose_rot)
        else:
            images = torch.stack(
                [self.image_transform(image) for image in images], dim=0)
            if self.train:
                if self.sparse:
                    # rotate and scale initalization targets
                    coords_w = math.ceil(images.size(3) / OUTPUT_SUBSAMPLE)
                    coords_h = math.ceil(images.size(2) / OUTPUT_SUBSAMPLE)
                    coords_set = []
                    for i in indexs:
                        if self.preload_image:
                            coords = self.loaded_scs[i]
                        else:
                            coords = torch.load(self.sc_imgs[i])
                            # (3,H,W)
                        coords = F.interpolate(coords.unsqueeze(0), size=(coords_h, coords_w))[0]

                        coords = coords.view(3, -1).permute(1, 0)
                        coords_set.append(coords)
                    frame['coords'] = torch.stack(coords_set, dim=0)
        
        T_tensor = torch.stack([cam_opt.torch_camera_pose_inv(pose[:3, :3], pose[:3, 3:4]) for pose in poses], dim=0)
        frame['T'] = T_tensor
        frame['K'] = K_tensor
        frame['image'] = images
        frame['indexs'] = indexs
        return frame

    def __len__(self):
        return len(self.samples)


def test():
    pass
  

