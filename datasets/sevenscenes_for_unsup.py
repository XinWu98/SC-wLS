import torch.utils.data as data
import numpy as np
from imageio import imread
import random
import os
import os.path as osp
import torch
import time
from torchvision import transforms
import torch.nn.functional as F
from skimage.transform import rotate, resize
from skimage import io
import math
import tools.camera_operator as cam_opt


OUTPUT_SUBSAMPLE = 8
# generate grid of target reprojection pixel positions
prediction_grid = torch.zeros((2,
                               math.ceil(1000 / OUTPUT_SUBSAMPLE),
                               # 1000px is max limit of image size, increase if needed
                               math.ceil(1000 / OUTPUT_SUBSAMPLE)))

# get the centre of 8*8 patch
for x in range(0, prediction_grid.size(2)):
    for y in range(0, prediction_grid.size(1)):
        prediction_grid[0, y, x] = x * OUTPUT_SUBSAMPLE + OUTPUT_SUBSAMPLE / 2
        prediction_grid[1, y, x] = y * OUTPUT_SUBSAMPLE + OUTPUT_SUBSAMPLE / 2


def load_as_float(path):
    return imread(path).astype(np.float32)


class SevenScenes(data.Dataset):
    def __init__(self, data_path, scene='chess', seed=7, train=True, transform=None, target_transform=None, real=False,
                 skip_images=False,
                 steps=1, keep_first=True,
                 skip=1, variable_skip=False,
                 augment=False,
                 aug_rotation=30,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_contrast=0.1,
                 aug_brightness=0.1,
                 image_height=480,
                 ):
        """
              :param real: If True, load poses from SLAM/integration of VO
              :param skip_images: If True, skip loading images and return None instead
        """
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

        self.scene = scene
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.train = train
        self.mean = torch.zeros((3))
        self.variable_skip = variable_skip
        self.steps = steps
        self.skip = skip
        self.keep_first = keep_first
        if self.steps == 1:
            self.skip = 0

        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)

        # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
           
        self.c_imgs = []
        self.sc_imgs = []
        self.seq_num = []
        self.seq_pose0 = []
        self.samples = []
        
        sequence_length = self.skip * (self.steps - 1)
        self.poses = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        gt_offset = int(0)
        def get_indices(index):
            if self.variable_skip:
                skips = np.random.randint(1, high=self.skip, size=self.steps - 1)
            else:
                skips = self.skip * np.ones(self.steps - 1)

            offsets = np.insert(skips, 0, 0).cumsum()
            # 20,10,0
            offsets = offsets.astype(np.int)[::-1]
            idx = index - offsets

            return idx

        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                           n.find('pose') >= 0]

            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                                       format(i))) for i in frame_idx]
            self.poses.extend(pss)
            self.seq_num.extend(np.ones_like(frame_idx) * seq)
            self.seq_pose0.append(pss[0])
         
            self.gt_idx = np.hstack((self.gt_idx, gt_offset + frame_idx))
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            if self.train:
                sc_imgs = [osp.join(data_path, '7scenes_init', '7scenes_{}'.format(scene), 'training', 'init',
                                    'seq{:02d}_frame-{:06d}.dat'.format(seq, i))
                           for i in frame_idx]

            self.c_imgs.extend(c_imgs)
            if self.train:
                self.sc_imgs.extend(sc_imgs)

            sample_set = [get_indices(i)+gt_offset for i in range(sequence_length, len(p_filenames))]
            self.samples.extend(sample_set)

            gt_offset += len(p_filenames)
        
        self.img_k = np.asarray([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)

        begin = time.time()

        self.loaded_imgs = []
        if self.train:
            self.loaded_scs = []

        self.pose_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        for idx, fn in enumerate(self.c_imgs):
            c_img = io.imread(fn)
            self.loaded_imgs.append(c_img)

            if self.train:
                coords = torch.load(self.sc_imgs[idx])
                assert coords is not None
                self.loaded_scs.append(coords)
        print("ReadData time:", time.time() - begin)

    def __getitem__(self, index):
        '''
        :return frame: a dict in torch form
        '''
        if self.skip_images:
            frame = None
        else:
            if self.c_imgs is None:
                print("NONE!")

            indexs = self.samples[index]
         
            frame = dict()
            poses = torch.stack([torch.from_numpy(self.poses[i]).float() for i in indexs], dim=0)
            K_tensor = torch.stack([torch.from_numpy(self.img_k.copy()) for i in indexs], dim=0)

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
              
                # rotate input image, don't change size(resize=False)
                def my_rot(t, angle, order, mode='constant'):
                    t = t.permute(1, 2, 0).numpy()
                    t = rotate(t, angle, order=order, mode=mode)
                    t = torch.from_numpy(t).permute(2, 0, 1).float()
                    return t
                images = torch.stack([my_rot(cur_image_transform(self.loaded_imgs[i]), angle, 1, 'reflect') for i in indexs],dim=0)
                               
                if self.train:
                    if self.sparse:
                        # rotate and scale initalization targets
                        coords_w = math.ceil(images.size(3) / OUTPUT_SUBSAMPLE)
                        coords_h = math.ceil(images.size(2) / OUTPUT_SUBSAMPLE)
                        coords_set = []
                        for i in indexs:
                            # (3,H,W)
                            coords = self.loaded_scs[i]  
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
                #Broadcast
                poses = torch.matmul(poses, pose_rot)

            else:
                images = torch.stack(
                    [self.image_transform(self.loaded_imgs[i]) for i in indexs], dim=0)
                if self.train:
                    if self.sparse:
                        # rotate and scale initalization targets
                        coords_w = math.ceil(images.size(3) / OUTPUT_SUBSAMPLE)
                        coords_h = math.ceil(images.size(2) / OUTPUT_SUBSAMPLE)
                        coords_set = []
                        for i in indexs:
                            # (3,H,W)
                            coords = self.loaded_scs[i] 
                            coords = F.interpolate(coords.unsqueeze(0), size=(coords_h, coords_w))[0]
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