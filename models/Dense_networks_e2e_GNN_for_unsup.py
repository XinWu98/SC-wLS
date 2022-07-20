import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from loss_functions import UnsupLoss, DLT_postprocess
from banet_track.ba_module import batched_pi_inv
from Dense_Nets.ESAC import ESAC_SCNet
from Dense_Nets.ESAC_DROID import ESAC_DROID_Net
from Dense_Nets.OAGNN_EigFree import OAGNN

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
        self.wDLT_net = OAGNN(self.config_e2e)           

        self.config = {
                'padding_mode':'zeros',
                'with_auto_mask':args.auto_mask,
                'with_ssim':args.ssim,
                'photo_weight':args.photo_weight,
                'use_DLT2T':args.use_DLT2T,
                'use_patch':args.use_patch,
                'Hc': 60,
                'Wc': 80
            }
        
        self.config_e2eloss = {}
        self.matchloss = UnsupLoss(self.config, self.config_e2eloss, self.relu)

    def forward(self, X_world, image, Ts, Ks, epoch, test=False):
        """
        :param X_world: corresponding 3D coord (B,T, H'W',3)
        :param image: (B,T, 3,H, W)
        :param Ts: (B,T, 3, 4)
        :param Ks: (B,T, 3, 3)
        :return: pred_world(B,3,H,W) , loss
        """
        if test==False:
            B, T, _, H, W = image.size()
            image = image.contiguous().view(-1,3,H,W)
            Ts = Ts.contiguous().view(-1,3,4)
            Ks = Ks.contiguous().view(-1,3,3)
        else:
            B, _, H, W = image.size()

        pred_X, _ = self.network(image)

        BT,_,cH,cW = pred_X.size()
        self.matchloss.config_un['Hc'] = cH
        self.matchloss.config_un['Wc'] = cW
        N = cH * cW
        pred_X_clone = pred_X.clone()
        pixel_grid_crop = pixel_grid[:, 0:pred_X.size(2), 0:pred_X.size(3)].clone()
        pixel_grid_crop = pixel_grid_crop.view(2, -1).permute(1,0).to(device)
        pixel_grid_crop_clone = pixel_grid_crop.clone()

        if test==True:                       
            # (B,N,3)
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

            return pred_X_clone, y_hat, e_post, pixel_grid_crop_clone
        else:
            image = image.view(B,T,3,H,W)            
            # (B,N,3)
            pred_X_seq = pred_X.detach().view(BT, 3, -1).permute(0, 2, 1) 
            pixel_grid_crop_seq = pixel_grid_crop.expand(BT,-1,-1)
            
            depth_norm = torch.ones(BT,N,1).to(device)
            Pc_norm = batched_pi_inv(Ks, pixel_grid_crop_seq, depth_norm)            
            sparse_data = torch.cat([Pc_norm[:,:,:2], pred_X_seq], dim=-1)
           
            res_logits,_ = self.wDLT_net(sparse_data,Ks)
            y_hat = res_logits[-1]

            loss_1 = 0
            loss_val = []
            e_hat = []
            num_valid=[]
            for i in range(len(res_logits)):
                loss_i, geo_loss, cla_loss, e_post, num_valid_sc = self.matchloss.run(epoch, sparse_data, res_logits[i], Ks, image, pixel_grid_crop_seq)
                loss_1 += loss_i
                loss_val += [geo_loss, cla_loss]
                e_hat.append(e_post)
                num_valid.append(num_valid_sc)
           
            loss_1 = torch.unsqueeze(loss_1, 0)
            loss_2 = torch.unsqueeze(loss_val[0], 0)
            loss_3 = torch.unsqueeze(loss_val[1], 0)

            if self.relu:
                y_hat = torch.relu(torch.tanh(y_hat))
            else:
                y_hat = torch.exp(F.logsigmoid(y_hat))

            pred_X_clone = pred_X_clone.view(B,T,3,cH, cW)
            y_hat = y_hat.view(B,T,-1)
            e_hat[-1] = e_hat[-1].view(B,T,3,4)
            return loss_1, loss_2, loss_3, pred_X_clone, y_hat, e_hat[-1], pixel_grid_crop_clone, num_valid[-1]
        
    def init_weights(self):
        self.network.init_weights()
   
