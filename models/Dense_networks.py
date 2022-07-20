import torch.nn as nn
import torch
import math
from loss_functions import dsacstar_loss
from Dense_Nets.ESAC import ESAC_SCNet
from Dense_Nets.ESAC_DROID import ESAC_DROID_Net

# generate grid of target reprojection pixel positions
OUTPUT_SUBSAMPLE = 8
pixel_grid = torch.zeros((2,
	math.ceil(5000 / OUTPUT_SUBSAMPLE),		# 5000px is max limit of image size, increase if needed
	math.ceil(5000 / OUTPUT_SUBSAMPLE)))

for x in range(0, pixel_grid.size(2)):
	for y in range(0, pixel_grid.size(1)):
		pixel_grid[0, y, x] = x * OUTPUT_SUBSAMPLE + OUTPUT_SUBSAMPLE / 2
		pixel_grid[1, y, x] = y * OUTPUT_SUBSAMPLE + OUTPUT_SUBSAMPLE / 2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class DenseSC(nn.Module):
    def __init__(self, args, scene_mean,e2e=False):
        super(DenseSC, self).__init__()
        self.mean = scene_mean.clone()
        self.e2e= e2e

        if args.uncertainty:
            self.un = True
        else:
            self.un = False
        self.dsacstar = args.dsacstar

        if args.dataset == '7Scenes':
            self.network = ESAC_SCNet(self.mean, self.un)
        else:
            self.network = ESAC_DROID_Net(self.mean, self.un, output_dim=256, norm_fn='none')
             
        if self.e2e==False:
            self.config = {
                    'itol': args.inittolerance,
                    'mindepth': args.mindepth,
                    'maxdepth': args.maxdepth,
                    'targetdepth': args.targetdepth,
                    'softclamp': args.softclamp,
                    'hardclamp': args.hardclamp
                }

    def forward(self, X_world, image, Ts, Ks):
        """
        :param X_world: corresponding 3D coord (B, H'W',3)
        :param used_mask: discard inaccurate depth (B,H'W')
        :param Ts: (B, 3, 4)
        :param Ks: (B, 3, 3)
        :return: pred_world(B,3,H,W) , loss
        """
        pred_X, uncertainty = self.network(image)
        pred_X_clone = pred_X.clone()
        pixel_grid_crop = pixel_grid[:, 0:pred_X.size(2), 0:pred_X.size(3)].clone()
        pixel_grid_crop = pixel_grid_crop.view(2, -1).permute(1,0).to(device)
        pixel_grid_crop_clone = pixel_grid_crop.clone()
       
        if self.dsacstar:
            loss_1, num_valid_sc = dsacstar_loss(pred_X, X_world, pixel_grid_crop, Ts, Ks, self.config)
            loss_1 = torch.unsqueeze(loss_1, 0)

            return loss_1, None, None, pred_X_clone, uncertainty, None, pixel_grid_crop_clone

        
    def init_weights(self):
        self.network.init_weights()
