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
                 augment=False,
                 preload_image=False,
                 aug_rotation=30,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_contrast=0.1,
                 aug_brightness=0.1,
                 aug_saturation=0.1,
                 image_height=480
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
        self.aug_saturation= aug_saturation
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_height),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                std=[0.25]
            )
        ])
    
        self.preload_image = preload_image

        self.scene = scene
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.train = train
        self.mean = torch.zeros((3))
        
        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)

        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
            
        self.c_imgs = []
        self.poses = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        gt_offset = int(0)
        self.names = []
        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            seq_name = 'seq-{:02d}'.format(seq)
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                           n.find('pose') >= 0]

            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                                       format(i))) for i in frame_idx]
            self.poses.extend(pss)
            self.gt_idx = np.hstack((self.gt_idx, gt_offset + frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            names = [osp.join(seq_name, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            self.names.extend(names)
            self.c_imgs.extend(c_imgs)
           
        self.img_k = np.asarray([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)

        begin = time.time()

        self.loaded_imgs = []
       
        self.pose_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if self.preload_image:
            for idx, fn in enumerate(self.c_imgs):
                c_img = io.imread(fn)
                self.loaded_imgs.append(c_img)

            print("ReadData time:", time.time() - begin)

    def __getitem__(self, index):
        '''
        :return frame: a dict in torch form
        '''
        if self.skip_images:
            frame = None
        else:
            if self.c_imgs is None:
                print("NONE!!!!!!!")
            frame = dict()
            pose = self.poses[index]
            pose = torch.from_numpy(pose).float()

            K = self.img_k.copy()
            K_tensor = torch.from_numpy(K) 

            if self.preload_image:
                image = self.loaded_imgs[index]
            else:
                image = io.imread(self.c_imgs[index])

            if self.augment:
                scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
                angle = random.uniform(-self.aug_rotation, self.aug_rotation)
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
               
                image = cur_image_transform(image)
                # scale focal length
                K_tensor[0,0] *= scale_factor
                K_tensor[1, 1] *= scale_factor
                # image center
                K_tensor[0, 2] = image.size(2)/2
                K_tensor[1, 2] = image.size(1)/2

                # rotate input image, don't change size(resize=False)
                def my_rot(t, angle, order, mode='constant'):
                    t = t.permute(1, 2, 0).numpy()
                    t = rotate(t, angle, order=order, mode=mode)
                    t = torch.from_numpy(t).permute(2, 0, 1).float()
                    return t
                #(3,H,W)
                image = my_rot(image, angle, 1, 'reflect')
                
                angle = angle * math.pi / 180
                pose_rot = torch.eye(4)
                pose_rot[0, 0] = math.cos(angle)
                pose_rot[0, 1] = -math.sin(angle)
                pose_rot[1, 0] = math.sin(angle)
                pose_rot[1, 1] = math.cos(angle)
                pose = torch.matmul(pose, pose_rot)
            else:
                image = self.image_transform(image)

            T_tensor = cam_opt.torch_camera_pose_inv(pose[:3,:3], pose[:3, 3:4])
            frame['T'] = T_tensor
            frame['K'] = K_tensor
            frame['image'] = image
            frame['name'] = self.names[index]
        return frame

    def __len__(self):
        return len(self.c_imgs)
