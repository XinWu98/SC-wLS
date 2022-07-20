from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from banet_track.ba_module import batched_pi, batched_inv_pose, batched_transpose, batched_pi_inv
from Dense_Nets.OAGNN_EigFree import batch_symeig
from reloc_pipeline.utils_func import compute_pnp_for_photometric

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CNN_OBJ_MAXINPUT = 1000.0 # reprojection errors are clamped at this magnitude
coord_softclamp = 10
coord_hardclamp = 1000
repro_softclamp = 10
repro_hardclamp = 100
min_uncertainty = 1e-5


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


def dsacstar_loss(pred_X, gt_X, pixel_grid_crop, T, K, config):
    """
    :param pred_X: B,3,H,W
    :param gt_X: B,N,3
    :param pixel_grid_crop: N,2
    :param T: B,3,4
    :param K: B,3,3
    :return:
    """
    R = T[:, :3, :3]
    t = T[:, :3, 3].squeeze(-1)

    pred_X = pred_X.view(T.size(0), 3, -1).permute(0,2,1)
    B, N, _ = pred_X.size()
    X_camera = batched_transpose(R, t, pred_X)   
    projected_2d, _ = batched_pi(K, X_camera)

    reprojection_error = projected_2d - pixel_grid_crop
    reprojection_error = reprojection_error.norm(2, 2)

    # check constraints
    invalid_min_depth = X_camera[:, :, 2] < config['mindepth']  
    invalid_repro = reprojection_error > config['hardclamp']  

    if gt_X is not None:
        gt_coords_mask = torch.abs(gt_X).sum(2) == 0

        target_camera = batched_transpose(R, t, gt_X)
        gt_coord_dist = torch.norm(X_camera - target_camera, dim=2, p=2)

        invalid_gt_distance = gt_coord_dist > config['itol'] 
        invalid_gt_distance[gt_coords_mask] = 0  

        valid_scene_coordinates = (invalid_min_depth + invalid_gt_distance + invalid_repro) == 0
    else:
        invalid_max_depth = X_camera[:, :, 2] > config['maxdepth']
        valid_scene_coordinates = (invalid_min_depth + invalid_max_depth + invalid_repro) == 0

    num_valid_sc = int(valid_scene_coordinates.sum())

    loss = 0

    if num_valid_sc > 0:
        
        reprojection_error = reprojection_error[valid_scene_coordinates]

        loss_l1 = reprojection_error[reprojection_error <= config['softclamp']]
        loss_sqrt = reprojection_error[reprojection_error > config['softclamp']]
        loss_sqrt = torch.sqrt(config['softclamp'] * loss_sqrt)

        loss += (loss_l1.sum() + loss_sqrt.sum())

    if num_valid_sc < B*N:

        invalid_scene_coordinates = (valid_scene_coordinates == 0)

        if gt_X is not None:
            invalid_scene_coordinates[gt_coords_mask] = 0
            loss += gt_coord_dist[invalid_scene_coordinates].sum()
        else:
            depth = config['targetdepth'] * torch.ones(B,N,1).to(device)
            pixel_grid_crop_batched = pixel_grid_crop.expand(B,-1,-1)
            target_camera = batched_pi_inv(K, pixel_grid_crop_batched, depth)

            loss += torch.abs(
                X_camera[invalid_scene_coordinates] - target_camera[invalid_scene_coordinates]).sum()

    loss /= B*N
    num_valid_sc /= B*N

    return loss, num_valid_sc



def reproj_label_norm(sparse_data, T, K, config):
    """
    Label the correspondence given gt_pose. 0 for inaccurate and unbelievable ones.
    :param pred_X: B,3,H,W
    :param gt_X: B,N,3
    :param pixel_grid_crop: N,2
    :param T: B,3,4
    :param K: B,3,3
    :return: is_pos, is_neg
    """
    R = T[:, :3, :3]
    t = T[:, :3, 3].squeeze(-1)

    pred_X = sparse_data[:,:,2:]
    Pc_norm = sparse_data[:,:,:2]
   
    X_camera = batched_transpose(R, t, pred_X)
    # check constraints
    invalid_min_depth = X_camera[:, :, 2] < config['mindepth']  
    invalid_max_depth = X_camera[:, :, 2] > config['maxdepth']

    X_camera[:,:,2].clamp_(1e-5)  # avoid division by zero
    X_camera_norm = X_camera[:,:,:2] / X_camera[:,:,2:3]
    
    reprojection_error = X_camera_norm - Pc_norm
    reprojection_error = reprojection_error.norm(2, 2)
    invalid_repro = reprojection_error > config['obj_geod_th']  

    is_pos = (invalid_min_depth + invalid_max_depth + invalid_repro) == 0
    is_neg = (is_pos == 0)

    return is_pos, is_neg

def cpu_svd(X):
    # it is much faster to run symeig on CPU
    X = X.cpu() 
    U,D,V_t = torch.svd(X)
    U = U.cuda()
    D = D.cuda()
    V_t = V_t.cuda() 
    return U,D,V_t

def cpu_det(X):
    # it is much faster to run symeig on CPU
    X = X.cpu() 
    value = torch.det(X)
    value = value.cuda() 
    return value

def cpu_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    _,v = torch.symeig(X, True)
    v = v.cuda()   
    return v

class EigFreeLoss(object):
    def __init__(self, config, relu=False):
        self.loss_eigfree = config['loss_eigfree']
        self.loss_classif = config['loss_classif']
        self.loss_essential_init_epo = config['loss_essential_init_epo']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.config = config
        self.relu = relu

    def run(self, global_step, data, logits, Ts, Ks):
        if self.relu:
            weights = torch.relu(torch.tanh(logits))
        else:
            # normalize neural guidance probabilities in log space, (-inf,0)		
            log_ng = F.logsigmoid(logits)       
            weights = torch.exp(log_ng)

        x_shp = data.shape

        # Make input data (num_img_pair x num_corr x 4), b*5*n, (u,v,x,y,z)
        xx = data.view(x_shp[0], x_shp[1], 5).permute(0, 2, 1)
        ones = torch.ones_like(xx[:, 2])
        zeros = torch.zeros_like(xx[:, 2])
        Xw = xx[:, 2]
        Yw = xx[:, 3]
        Zw = xx[:, 4]
        u = xx[:, 0]
        v = xx[:, 1]

        X_up = torch.stack([
            Xw, Yw, Zw, ones,
            zeros, zeros, zeros, zeros,
            -u * Xw, -u * Yw, -u*Zw , -u
        ], dim=1).permute(0, 2, 1)
        X_bottom = torch.stack([
            zeros, zeros, zeros, zeros,
            Xw, Yw, Zw, ones,
            -v * Xw, -v * Yw, -v * Zw, -v
        ], dim=1).permute(0, 2, 1)
        X = torch.cat([X_up, X_bottom], dim=1)
        
        w_cat = torch.cat([weights, weights], dim=-1)
        wX = w_cat.unsqueeze(-1) * X
        
        XwX = torch.matmul(X.permute(0, 2, 1), wX)

        e_gt = Ts.view(-1, 12)
        e_gt  = e_gt / (torch.norm(e_gt,dim=1,p=2).unsqueeze(-1))
        e_gt = e_gt.unsqueeze(-1)
        e_gt_t = e_gt.permute(0,2,1)

        d_term = torch.matmul(torch.matmul(e_gt_t, XwX), e_gt)
        
        e_hat = torch.eye(12).expand(x_shp[0],-1,-1).to(device) - torch.matmul(e_gt, e_gt_t)
        e_hat_t = e_hat.permute(0,2,1)
        XwX_e_neg = torch.matmul(torch.matmul(e_hat_t, XwX), e_hat)
        r_term = torch.einsum('bii->b', XwX_e_neg)
        eigenfree_loss = torch.mean(d_term + self.alpha*torch.exp(-self.beta*r_term))
        
        # Classification loss
        is_pos, is_neg = reproj_label_norm(data, Ts, Ks, self.config)

        is_pos = is_pos.type(logits.type())
        is_neg = is_neg.type(logits.type())
        c = is_pos - is_neg
        if self.relu:
            classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
        else:
            classif_losses = -F.logsigmoid(c * logits)
        
        # balance
        num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
        num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
        classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
        classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)  
        classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)
       
        loss = 0
        # Check global_step and add essential loss
        if self.loss_eigfree > 0 and global_step >= self.loss_essential_init_epo:
            loss += self.loss_eigfree * eigenfree_loss
        if self.loss_classif > 0:
            loss += self.loss_classif * classif_loss
        
        with torch.no_grad():
            # v = batch_symeig(XwX)
            _, v = torch.linalg.eigh(XwX, UPLO='U')
            e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 3, 4))
           
            R_hat = e_hat[:,:,:3]
            t_hat = e_hat[:,:,3]

            # U,D,V = torch.svd(R_hat) 
            # U,D,V = cpu_svd(R_hat)
            U, D, Vh = torch.linalg.svd(R_hat, full_matrices=False)
            c = torch.reciprocal(D.sum(dim=-1)/3.0).unsqueeze(-1)
           
            # Verify the sign of c, using the point with highest weight
            _, indice = torch.max(weights, dim=-1)
            Point_verify = xx[torch.arange(x_shp[0]), 2:, indice]
            ones = torch.ones([x_shp[0], 1]).to(device)
            Point_verify = torch.cat([Point_verify, ones], dim=-1).unsqueeze(1)
        
            criterion = c * (torch.matmul(Point_verify, e_hat[:,-1].view(x_shp[0], 4, 1)).squeeze(-1))
           
            inv_c = -1*c
            c = torch.where(criterion > 0, c, inv_c)
           
            R_hat = torch.matmul(U, Vh.conj()) * torch.sign(c.unsqueeze(-1))
            t_hat = c * t_hat
           
            #(b,3,4)
            e_post = torch.cat([R_hat, t_hat.unsqueeze(-1)], dim=-1)

        return [loss, eigenfree_loss, classif_loss, e_post, [d_term.mean(), r_term.mean(), eigenfree_loss]]


def mean_on_mask(diff, valid_mask):
    """
    :param diff: B,3,H',W'
    :param valid_mask: B,H',W'
    :return:
    """
    mask = valid_mask.unsqueeze(1)
    mean_value = (diff * mask).sum() / (mask.sum()*3)
    return mean_value

def patch_photometric_repro_loss(pred_X, image, pixel_grid_crop, T, K, config):
    """
    :param pred_X: BT,N,3
    :param image: B,T,3,H,W
    :param pixel_grid_crop: BT,N,2
    :param T: BT,3,4
    :param K: BT,3,3
    :return:
    """
    BT,N,_ = pred_X.size()
    B,T_num,_,H,W = image.size()
    Hc = config['Hc']
    Wc = config['Wc']

    R = T[:, :3, :3]
    t = T[:, :3, 3].squeeze(-1)
    pixel_grid_crop = pixel_grid_crop[0]

    X_camera = batched_transpose(R, t, pred_X)
    _, projected_depth = batched_pi(K, X_camera)
   
    if T_num == 1:
        return None, 0
    #photometric loss
    photo_loss = torch.tensor([0.0], requires_grad=True).to(device)
    
    R = R.view(B,T_num,3,3)
    t = t.view(B,T_num,3)
    pred_X = pred_X.view(B,T_num,-1,3)
    K = K.view(B,T_num,3,3)
  
    #use patch
    projected_depth = projected_depth.view(B,T_num,-1,1)

    meshgrid = np.meshgrid([-2, 0, 2], [-2, 0, 2], indexing='xy')
    meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
    meshgrid = torch.Tensor(meshgrid).to(device).permute(1, 2, 0).view(1, 1, 9, 2)

    tgt_points = pixel_grid_crop.expand(B,N,2).unsqueeze(2) + meshgrid 
    tgt_points = tgt_points.contiguous().view(B,-1,2)

    tgt_X_norm = 2 * tgt_points[..., 0] / (W - 1) - 1
    tgt_Y_norm = 2 * tgt_points[..., 1] / (H - 1) - 1
    H3 = 3*Hc
    W3 = 3*Wc

    tgt_pixel_coords = torch.stack([tgt_X_norm, tgt_Y_norm], dim=2).view(B,Hc,Wc,3,3,2).permute(0,1,3,2,4,5).contiguous().view(B,H3,W3,2)

    valid_masks = torch.zeros([B,1,Hc,Wc]).to(device)
    num_valid_sc = 0
    for i in range(T_num-1):
        src_img = image[:,i]
        tgt_img = image[:,i+1]
        tgt_depth = projected_depth[:,i+1].expand(-1, -1, 9).contiguous().view(B,-1,1)

        src_R = R[:,i]
        src_t = t[:,i]
        src_K = K[:,i]
        tgt_Rinv,tgt_tinv = batched_inv_pose(R[:,i+1], t[:,i+1])

        tgt_X_camera = batched_pi_inv(src_K, tgt_points, tgt_depth)
        tgt_pred_X = batched_transpose(tgt_Rinv, tgt_tinv, tgt_X_camera)
        src_X_camera = batched_transpose(src_R, src_t, tgt_pred_X)
        
        src_projected_2d, _ = batched_pi(src_K, src_X_camera)
       
        # Normalized
        X_norm = 2 * src_projected_2d[:,:,0] / (W - 1) - 1
        Y_norm = 2 * (src_projected_2d[:,:,1]) / (H - 1) - 1

        if config['padding_mode'] == 'zeros':
            X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
            X_norm[X_mask] = 2
            Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
            Y_norm[Y_mask] = 2
        src_pixel_coords = torch.stack([X_norm, Y_norm], dim=2).view(B,Hc,Wc,3,3,2).permute(0,1,3,2,4,5).contiguous().view(B,H3,W3,2)  
        projected_img = F.grid_sample(src_img, src_pixel_coords, padding_mode=config['padding_mode'])
        tgt_img_scaled = F.grid_sample(tgt_img, tgt_pixel_coords,padding_mode=config['padding_mode'])

        valid_points = (src_pixel_coords.abs().max(dim=-1)[0] <= 1)
        valid_mask = valid_points.float()
        
        diff_img = (tgt_img_scaled - projected_img).abs().clamp(0, 1)
        if config['with_auto_mask'] == True:
            src_img_scaled=F.grid_sample(src_img, tgt_pixel_coords,padding_mode=config['padding_mode'])
            auto_mask = (diff_img.mean(dim=1, keepdim=False) < (tgt_img_scaled - src_img_scaled).abs().mean(dim=1,
                                                                                             keepdim=False)).float() * valid_mask
            valid_mask = auto_mask

        if config['with_ssim'] == True:
            ssim_map = compute_ssim_loss(tgt_img_scaled, projected_img)
            diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        if valid_mask.sum() > 0:
            photo_loss += mean_on_mask(diff_img, valid_mask)
        num_valid_sc += valid_mask.sum()
        valid_masks = torch.cat((valid_masks, valid_mask[:,1::3,1::3].unsqueeze(1)), dim=1)
    valid_masks = valid_masks.view(BT,-1)
    photo_loss /= T_num - 1

    total_loss = config['photo_weight'] * photo_loss

    num_valid_sc  /= B*(T_num-1)*N

    return total_loss, num_valid_sc

def photometric_repro_loss(pred_X, image, pixel_grid_crop, T, K, config):
    """
    :param pred_X: BT,N,3
    :param gt_X: None
    :param image: B,T,3,H,W
    :param pixel_grid_crop: B,N,2
    :param T: BT,3,4
    :param K: BT,3,3
    :return:
    """
    BT,N,_ = pred_X.size()
    B,T_num,_,H,W = image.size()
    Hc = config['Hc']
    Wc = config['Wc']

    R = T[:, :3, :3]
    t = T[:, :3, 3].squeeze(-1)
    pixel_grid_crop = pixel_grid_crop[0]
   
    if T_num == 1:
        return None, 0
    #photometric loss
    photo_loss = torch.tensor([0.0], requires_grad=True).to(device)
    R = R.view(B,T_num,3,3)
    t = t.view(B,T_num,3)
    pred_X = pred_X.view(B,T_num,-1,3)
    K = K.view(B,T_num,3,3)

    src_X_norm = 2 * pixel_grid_crop[:, 0] / (W - 1) - 1
    src_Y_norm = 2 * pixel_grid_crop[:, 1] / (H - 1) - 1
    tgt_pixel_coords = torch.stack([src_X_norm, src_Y_norm], dim=1).view(Hc, Wc, 2).unsqueeze(0)
    tgt_pixel_coords = tgt_pixel_coords.expand(B,Hc,Wc,2)

    valid_masks = torch.zeros([B,1,Hc,Wc]).to(device)
    num_valid_sc = 0
    for i in range(T_num-1):
        src_img = image[:,i]
        tgt_img = image[:,i+1]
        tgt_pred_X=  pred_X[:,i+1]
        src_R = R[:,i]
        src_t = t[:,i]
        src_K = K[:,i]
        src_X_camera = batched_transpose(src_R, src_t, tgt_pred_X)
        src_projected_2d, _ = batched_pi(src_K, src_X_camera)

        X_norm = 2 * src_projected_2d[:,:,0] / (W - 1) - 1
        Y_norm = 2 * (src_projected_2d[:,:,1]) / (H - 1) - 1  

        if config['padding_mode'] == 'zeros':
            X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
            X_norm[X_mask] = 2
            Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
            Y_norm[Y_mask] = 2
        src_pixel_coords = torch.stack([X_norm, Y_norm], dim=2).view(B,Hc,Wc,2)  
        projected_img = F.grid_sample(src_img, src_pixel_coords, padding_mode=config['padding_mode'])
        tgt_img_scaled = F.grid_sample(tgt_img, tgt_pixel_coords,padding_mode=config['padding_mode'])

        valid_points = (src_pixel_coords.abs().max(dim=-1)[0] <= 1)
        valid_mask = valid_points.float()
        
        diff_img = (tgt_img_scaled - projected_img).abs().clamp(0, 1)
        if config['with_auto_mask'] == True:
            src_img_scaled=F.grid_sample(src_img, tgt_pixel_coords,padding_mode=config['padding_mode'])
            auto_mask = (diff_img.mean(dim=1, keepdim=False) < (tgt_img_scaled - src_img_scaled).abs().mean(dim=1,
                                                                                             keepdim=False)).float() * valid_mask
            valid_mask = auto_mask

        if config['with_ssim'] == True:
            ssim_map = compute_ssim_loss(tgt_img_scaled, projected_img)
            diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        if valid_mask.sum() > 0:
            photo_loss += mean_on_mask(diff_img, valid_mask)
        num_valid_sc += valid_mask.sum()
        valid_masks = torch.cat((valid_masks, valid_mask.unsqueeze(1)), dim=1)
    valid_masks = valid_masks.view(BT,-1)
    photo_loss /= T_num - 1

    total_loss = config['photo_weight'] * photo_loss

    num_valid_sc  /= B*(T_num-1)*N
    
    return total_loss, num_valid_sc

def DLT_postprocess(data, logits, Ks, relu):
    if relu:
        weights = torch.relu(torch.tanh(logits))
    else:
        log_ng = F.logsigmoid(logits )       
        weights = torch.exp(log_ng)
    
    x_shp = data.shape
    
    # Make input data (num_img_pair x num_corr x 4), b*5*n, (u,v,x,y,z)
    xx = data.view(x_shp[0], x_shp[1], 5).permute(0, 2, 1)
    ones = torch.ones_like(xx[:, 2])
    zeros = torch.zeros_like(xx[:, 2])
    Xw = xx[:, 2]
    Yw = xx[:, 3]
    Zw = xx[:, 4]
    u = xx[:, 0]
    v = xx[:, 1]

    X_up = torch.stack([
        Xw, Yw, Zw, ones,
        zeros, zeros, zeros, zeros,
        -u * Xw, -u * Yw, -u*Zw , -u
    ], dim=1).permute(0, 2, 1)
    X_bottom = torch.stack([
        zeros, zeros, zeros, zeros,
        Xw, Yw, Zw, ones,
        -v * Xw, -v * Yw, -v * Zw, -v
    ], dim=1).permute(0, 2, 1)
   
    X = torch.cat([X_up, X_bottom], dim=1)
    
    w_cat = torch.cat([weights, weights], dim=-1)
    wX = w_cat.unsqueeze(-1) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    _, v = torch.linalg.eigh(XwX, UPLO='U')
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 3, 4))
    R_hat = e_hat[:,:,:3]
    t_hat = e_hat[:,:,3]

    # U,D,V = torch.svd(R_hat)
    U, D, Vh = torch.linalg.svd(R_hat, full_matrices=False)
    c = torch.reciprocal(D.sum(dim=-1)/3.0).unsqueeze(-1)
    
    # Verify the sign of c, using the point with highest weight
    _, indice = torch.max(weights, dim=-1)
    Point_verify = xx[torch.arange(x_shp[0]), 2:, indice]
    ones = torch.ones([x_shp[0], 1]).to(device)
    Point_verify = torch.cat([Point_verify, ones], dim=-1).unsqueeze(1)

    criterion = c * (torch.matmul(Point_verify, e_hat[:,-1].view(x_shp[0], 4, 1)).squeeze(-1))
    inv_c = -1*c
    c = torch.where(criterion > 0, c, inv_c)
    
    R_hat = torch.matmul(U, Vh.conj()) * torch.sign(c.unsqueeze(-1))
    t_hat = c * t_hat
    
    #(b,3,4)
    e_post = torch.cat([R_hat, t_hat.unsqueeze(-1)], dim=-1)
    return e_post

class UnsupLoss(object):
    def __init__(self, config_un,config, relu):
        self.config = config
        self.config_un = config_un
        self.relu = relu

    def run(self, global_step, data, logits, Ks, image, pixel_grid_crop):
        if self.relu:
            weights = torch.relu(torch.tanh(logits))
        else:
            log_ng = F.logsigmoid(logits )       
            weights = torch.exp(log_ng)
        x_shp = data.shape

        if self.config_un['use_DLT2T']:
            # Make input data (num_img_pair x num_corr x 4), b*5*n, (u,v,x,y,z)
            xx = data.view(x_shp[0], x_shp[1], 5).permute(0, 2, 1)
            ones = torch.ones_like(xx[:, 2])
            zeros = torch.zeros_like(xx[:, 2])
            Xw = xx[:, 2]
            Yw = xx[:, 3]
            Zw = xx[:, 4]
            u = xx[:, 0]
            v = xx[:, 1]

            X_up = torch.stack([
                Xw, Yw, Zw, ones,
                zeros, zeros, zeros, zeros,
                -u * Xw, -u * Yw, -u*Zw , -u
            ], dim=1).permute(0, 2, 1)
            X_bottom = torch.stack([
                zeros, zeros, zeros, zeros,
                Xw, Yw, Zw, ones,
                -v * Xw, -v * Yw, -v * Zw, -v
            ], dim=1).permute(0, 2, 1)
            
            X = torch.cat([X_up, X_bottom], dim=1)

            w_cat = torch.cat([weights, weights], dim=-1)
            wX = w_cat.unsqueeze(-1) * X
           
            XwX = torch.matmul(X.permute(0, 2, 1), wX)

            _, v = torch.linalg.eigh(XwX, UPLO='U')
            e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 3, 4))
            R_hat = e_hat[:,:,:3]
            t_hat = e_hat[:,:,3]

            # U,D,V = torch.svd(R_hat)
            U, D, Vh = torch.linalg.svd(R_hat, full_matrices=False)
            c = torch.reciprocal(D.sum(dim=-1)/3.0).unsqueeze(-1)
            
            # Verify the sign of c, using the point with highest weight
            _, indice = torch.max(weights, dim=-1)
            Point_verify = xx[torch.arange(x_shp[0]), 2:, indice]
            ones = torch.ones([x_shp[0], 1]).to(device)
            Point_verify = torch.cat([Point_verify, ones], dim=-1).unsqueeze(1)
            criterion = c * (torch.matmul(Point_verify, e_hat[:,-1].view(x_shp[0], 4, 1)).squeeze(-1))
            inv_c = -1*c
            c = torch.where(criterion > 0, c, inv_c)
            
            R_hat = torch.matmul(U, Vh.conj()) * torch.sign(c.unsqueeze(-1))
            t_hat = c * t_hat
            
            #(b,3,4)
            e_post = torch.cat([R_hat, t_hat.unsqueeze(-1)], dim=-1)
            e_post_DLT = e_post
        else:
            with torch.no_grad():
                # Make input data (num_img_pair x num_corr x 4), b*5*n, (u,v,x,y,z)
                xx = data.view(x_shp[0], x_shp[1], 5).permute(0, 2, 1)
                ones = torch.ones_like(xx[:, 2])
                zeros = torch.zeros_like(xx[:, 2])
                Xw = xx[:, 2]
                Yw = xx[:, 3]
                Zw = xx[:, 4]
                u = xx[:, 0]
                v = xx[:, 1]

                X_up = torch.stack([
                    Xw, Yw, Zw, ones,
                    zeros, zeros, zeros, zeros,
                    -u * Xw, -u * Yw, -u*Zw , -u
                ], dim=1).permute(0, 2, 1)
                X_bottom = torch.stack([
                    zeros, zeros, zeros, zeros,
                    Xw, Yw, Zw, ones,
                    -v * Xw, -v * Yw, -v * Zw, -v
                ], dim=1).permute(0, 2, 1)
                
                X = torch.cat([X_up, X_bottom], dim=1)
                w_cat = torch.cat([weights, weights], dim=-1)
                wX = w_cat.unsqueeze(-1) * X
                XwX = torch.matmul(X.permute(0, 2, 1), wX)
                _, v = torch.linalg.eigh(XwX, UPLO='U')
                e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 3, 4))
                R_hat = e_hat[:,:,:3]
                t_hat = e_hat[:,:,3]

                U, D, Vh = torch.linalg.svd(R_hat, full_matrices=False)
                c = torch.reciprocal(D.sum(dim=-1)/3.0).unsqueeze(-1)
               
                _, indice = torch.max(weights, dim=-1)
                Point_verify = xx[torch.arange(x_shp[0]), 2:, indice]
                ones = torch.ones([x_shp[0], 1]).to(device)
                Point_verify = torch.cat([Point_verify, ones], dim=-1).unsqueeze(1)
                criterion = c * (torch.matmul(Point_verify, e_hat[:,-1].view(x_shp[0], 4, 1)).squeeze(-1))
                inv_c = -1*c
                c = torch.where(criterion > 0, c, inv_c)
                
                R_hat = torch.matmul(U, Vh.conj()) * torch.sign(c.unsqueeze(-1))
                t_hat = c * t_hat
                
                #(b,3,4)
                e_post_DLT = torch.cat([R_hat, t_hat.unsqueeze(-1)], dim=-1)

                e_post = compute_pnp_for_photometric(data[:,:,2:], Ks, data[:,:,:2])
                e_post = e_post.to(device)
            
        if self.config_un['use_patch']:
            loss_1, num_valid_sc = patch_photometric_repro_loss(data[:,:,2:], image, pixel_grid_crop, e_post, Ks,
                                                                self.config_un)

        else:
            loss_1, num_valid_sc = photometric_repro_loss(data[:,:,2:], image, pixel_grid_crop, e_post, Ks,
                                                                self.config_un)


        return [loss_1, loss_1, loss_1, e_post_DLT, num_valid_sc]


