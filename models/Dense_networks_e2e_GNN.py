import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from loss_functions import EigFreeLoss, DLT_postprocess
from banet_track.ba_module import batched_pi_inv
from Dense_Nets.ESAC import ESAC_SCNet
from Dense_Nets.ESAC_DROID import ESAC_DROID_Net
from Dense_Nets.OAGNN_EigFree import OAGNN, OANet

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

class DenseSC_GNN(nn.Module):
    def __init__(self, args, scene_mean):
        super(DenseSC_GNN, self).__init__()
        self.mean = scene_mean.clone()
        self.e2e_training = args.e2e_training

        if args.uncertainty:
            self.un = True
        else:
            self.un = False
        if args.dataset == '7Scenes':
            self.relu = True
        else:
            self.relu = False

        if args.dataset == '7Scenes':
            self.network = ESAC_SCNet(self.mean, self.un)
        else:
            self.network = ESAC_DROID_Net(self.mean, self.un, output_dim=256, norm_fn='none')
             
        self.config_e2e = {
                'net_depth': args.net_depth,
                'clusters': args.clusters,
                'iter_num': args.iter_num,
                'net_channels': args.net_channels
            }
        # GNN, self-attention
        self.wDLT_net = OAGNN(self.config_e2e)

        # self.wDLT_net = OANet(self.config_e2e)
   
        self.config = {
                'itol': args.inittolerance,
                'mindepth': args.mindepth,
                'maxdepth': args.maxdepth,
                'targetdepth': args.targetdepth,
                'softclamp': args.softclamp,
                'hardclamp': args.hardclamp
            }
        self.config_e2eloss = {
            "loss_eigfree" : args.loss_essential,
            "loss_classif" : args.loss_classif,
            "loss_essential_init_epo": args.loss_essential_init_epo,
            'mindepth': args.mindepth,
            'maxdepth': args.maxdepth,
            'obj_geod_th': args.obj_geod_th,
            'alpha': args.Eig_alpha,
            'beta': args.Eig_beta,
        }
        self.matchloss = EigFreeLoss(self.config_e2eloss, self.relu)


    def forward(self, X_world, image, Ts, Ks, epoch, test=False, ds=False):
        """
        :param X_world: corresponding 3D coord (B, H'W',3)
        :param used_mask: discard inaccurate depth (B,H'W')
        :param Ts: (B, 3, 4)
        :param Ks: (B, 3, 3)
        :return: pred_world(B,3,H,W) , loss
        """
        # B*3*H*W 
        pred_X, _ = self.network(image)
        B,_,cH,cW = pred_X.size()
        N = cH * cW
              
        #(2,H,W)
        pixel_grid_crop = pixel_grid[:, 0:pred_X.size(2), 0:pred_X.size(3)].clone()
        #(N,2)
        pixel_grid_crop = pixel_grid_crop.view(2, -1).permute(1,0).to(device)
        if ds==True:
            return pred_X, None, None, pixel_grid_crop
  
        if test==True:
            pred_X_seq = pred_X.detach().view(B, 3, -1).permute(0, 2, 1)
            pixel_grid_crop_seq = pixel_grid_crop.expand(B,-1,-1)
            
            depth_norm = torch.ones(B,N,1).to(device)
            Pc_norm = batched_pi_inv(Ks, pixel_grid_crop_seq, depth_norm)            
            sparse_data = torch.cat([Pc_norm[:,:,:2], pred_X_seq], dim=-1)
          
            res_logits,_ = self.wDLT_net(sparse_data,Ks)
            y_hat = res_logits[-1]

            e_post = DLT_postprocess(sparse_data, res_logits[-1], Ks, self.relu)

            if self.relu:
                y_hat = torch.relu(torch.tanh(y_hat))
            else:
                y_hat = torch.exp(F.logsigmoid(y_hat))
            return pred_X, y_hat, e_post, pixel_grid_crop
        else:
            pred_X_clone = pred_X.clone()
            pixel_grid_crop_clone = pixel_grid_crop.clone()
            # (B,N,3)            
            if self.e2e_training:
                pred_X_seq = pred_X.view(B, 3, -1).permute(0, 2, 1)
            else:
                pred_X_seq = pred_X.detach().view(B, 3, -1).permute(0, 2, 1)

            pixel_grid_crop_seq = pixel_grid_crop.expand(B,-1,-1)
            
            depth_norm = torch.ones(B,N,1).to(device)
            Pc_norm = batched_pi_inv(Ks, pixel_grid_crop_seq, depth_norm)            
            sparse_data = torch.cat([Pc_norm[:,:,:2], pred_X_seq], dim=-1)
           
            res_logits,_ = self.wDLT_net(sparse_data,Ks)
            y_hat = res_logits[-1]
            
            loss_1 = 0
            loss_val = []
            e_hat = []
            dr_term=[]
            for i in range(len(res_logits)):
                loss_i, geo_loss, cla_loss, e_post, dr = self.matchloss.run(epoch, sparse_data, res_logits[i], Ts, Ks)
                loss_1 += loss_i
                loss_val += [geo_loss, cla_loss]
                e_hat.append(e_post)
                dr_term.append(dr)
          
            loss_1 = torch.unsqueeze(loss_1, 0)
            loss_2 = torch.unsqueeze(loss_val[0], 0)
            loss_3 = torch.unsqueeze(loss_val[1], 0)

            if self.relu:
                y_hat = torch.relu(torch.tanh(y_hat))
            else:
                y_hat = torch.exp(F.logsigmoid(y_hat))

            return loss_1, loss_2, loss_3, pred_X_clone, y_hat, e_hat[-1], pixel_grid_crop_clone, dr_term[-1][0], dr_term[-1][1]



    def init_weights(self):
        self.network.init_weights()
   
