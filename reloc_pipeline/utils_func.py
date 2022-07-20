import cv2
import numpy as np
import torch
from evaluator.basic_metric import rel_t, rel_R


def compute_pnp_for_photometric(query_X_w, query_Ks, pnp_x_2ds, repro_thres=10):
    """
    :param query_X_w: BT,N,3
    :param query_Ks: (B,3,3)
    :param pnp_x_2ds: (BT,N,2)
    :param repro_thres:
    :return:
    """
    lm_pnp_poses = []
    for idx, query_X_3d_w in enumerate(query_X_w):
        # (N)
        query_K = query_Ks[idx, :, :].squeeze(0)
        pnp_x_2d = pnp_x_2ds[idx]

        _, R_vec, t, inliers = cv2.solvePnPRansac(query_X_3d_w.cpu().detach().numpy().reshape(-1, 1, 3),
                                                  pnp_x_2d.cpu().detach().numpy().reshape(-1, 1, 2),
                                                  query_K.cpu().detach().numpy(),
                                                  None,
                                                  useExtrinsicGuess=False,
                                                  reprojectionError=10.0,
                                                  iterationsCount=128,
                                                  flags=cv2.SOLVEPNP_P3P
                                                  # minInliersCount=200
                                                  )

        R_res, _ = cv2.Rodrigues(R_vec)
        lm_pnp_pose = np.eye(4, dtype=np.float32)
        lm_pnp_pose[:3, :3] = R_res
        lm_pnp_pose[:3, 3] = t.ravel()
       
        lm_pnp_poses.append(torch.Tensor(lm_pnp_pose[:3,:]))
        
    return torch.stack(lm_pnp_poses, dim=0)

    
def compute_pose_lm_pnp(gt_Ts, query_X_w, query_Ks, pnp_x_2ds, scene_valid_mask=None, uncertaintys=None, repro_thres=10):
    """
    :param gt_Ts: (B,3,4)
    :param query_X_w: B,3,H,W / B,3,N
    :param query_Ks: (B,3,3)
    :param pnp_x_2ds: (B,N,2)
    :param scene_valid_mask:(B,N)
    :param uncertainty:B,1,H,W
    :param repro_thres:
    :return:
    """
    lm_pnp_poses = []
    R_accs = []
    t_accs = []
   
    for idx, query_X_3d_w in enumerate(query_X_w):
        # (N)
        gt_T = gt_Ts[idx, :, :].squeeze(0)
        query_K = query_Ks[idx, :, :].squeeze(0)
        pnp_x_2d = pnp_x_2ds[0, :, :]
        query_X_3d_w = query_X_3d_w.view(3,-1).permute(1,0)
        if uncertaintys is not None:
            uncertainty = uncertaintys[idx].view(-1)
        if scene_valid_mask is not None:
            mask = scene_valid_mask[idx, :]  
            query_X_3d_w = query_X_3d_w[mask]
            pnp_x_2d = pnp_x_2d[mask]
            if uncertaintys is not None:
                uncertainty = uncertainty[mask]

        if uncertaintys is not None:
            # 2400:deciding #of used points
            sample_num = min(2400, pnp_x_2d.size(0))
            sample_id = torch.argsort(uncertainty, descending=True)[:sample_num]

            # sample_id=torch.arange(0,uncertainty.numel())[uncertainty.ge(0.9)]
        
            query_X_3d_w = query_X_3d_w[sample_id]
            pnp_x_2d = pnp_x_2d[sample_id]

            # Or use thresh of weight to decide.
            # sample_id = np.argwhere(uncertaintys.cpu().numpy() > 0.5)
            # query_X_3d_w = query_X_3d_w.cpu()[sample_id]
            # pnp_x_2d = pnp_x_2d.cpu()[sample_id]
     
        # reshape must be consistent with 3D Coords's order(3,N)
        _, R_vec, t, inliers = cv2.solvePnPRansac(query_X_3d_w.cpu().detach().numpy().reshape(-1, 1, 3),
                                                  pnp_x_2d.cpu().detach().numpy().reshape(-1, 1, 2),
                                                  query_K.cpu().detach().numpy(),
                                                  None,
                                                  useExtrinsicGuess=False,
                                                  reprojectionError=10,
                                                  iterationsCount=128,
                                                  confidence=0.99,
                                                  flags=cv2.SOLVEPNP_P3P
                                                  # minInliersCount=200
                                                  )

        R_res, _ = cv2.Rodrigues(R_vec)
        lm_pnp_pose = np.eye(4, dtype=np.float32)
        lm_pnp_pose[:3, :3] = R_res
        lm_pnp_pose[:3, 3] = t.ravel()

        # measure accuracy
        gt_pose = gt_T.detach().cpu().numpy()
        R_acc = rel_R(lm_pnp_pose, gt_pose)
        t_acc = rel_t(lm_pnp_pose, gt_pose)
        lm_pnp_poses.append(torch.Tensor(lm_pnp_pose[:3,:]))
        R_accs.append(R_acc)
        t_accs.append(t_acc)
        
    return R_accs, t_accs, lm_pnp_poses

def compute_pose_only(gt_Ts, query_X_w, query_Ks, pnp_x_2ds, scene_valid_mask=None, uncertaintys=None, repro_thres=10):
    """
    :param gt_Ts: (B,3,4)
    :param query_X_w: B,3,H,W / B,3,N
    :param query_Ks: (B,3,3)
    :param pnp_x_2ds: (B,N,2)
    :param scene_valid_mask:(B,N)
    :param uncertainty:B,1,H,W
    :param repro_thres:
    :return:
    """
    lm_pnp_poses = []
    
    for idx, query_X_3d_w in enumerate(query_X_w):
        # (N)
        gt_T = gt_Ts[idx, :, :].squeeze(0)
        query_K = query_Ks[idx, :, :].squeeze(0)
        pnp_x_2d = pnp_x_2ds[0, :, :]
        query_X_3d_w = query_X_3d_w.view(3,-1).permute(1,0)
        if uncertaintys is not None:
            uncertainty = uncertaintys[idx].view(-1)
        if scene_valid_mask is not None:
            mask = scene_valid_mask[idx, :] 
            query_X_3d_w = query_X_3d_w[mask]
            pnp_x_2d = pnp_x_2d[mask]
            if uncertaintys is not None:
                uncertainty = uncertainty[mask]

        if uncertaintys is not None:
            # 2400:deciding #of used points
            sample_num = min(2400, pnp_x_2d.size(0))
            sample_id = torch.argsort(uncertainty, descending=True)[:sample_num]

            # sample_id=torch.arange(0,uncertainty.numel())[uncertainty.ge(0.9)]
           
            query_X_3d_w = query_X_3d_w[sample_id]
            pnp_x_2d = pnp_x_2d[sample_id]

        # reshape must be consistent with 3D Coords's order(3,N)
        _, R_vec, t, inliers = cv2.solvePnPRansac(query_X_3d_w.cpu().detach().numpy().reshape(-1, 1, 3),
                                                  pnp_x_2d.cpu().detach().numpy().reshape(-1, 1, 2),
                                                  query_K.cpu().detach().numpy(),
                                                  None,
                                                  useExtrinsicGuess=False,
                                                  reprojectionError=10,
                                                  iterationsCount=128,
                                                  confidence=0.99,
                                                  flags=cv2.SOLVEPNP_P3P
                                                  # minInliersCount=200
                                                  )

        R_res, _ = cv2.Rodrigues(R_vec)
        lm_pnp_pose = np.eye(4, dtype=np.float32)
        lm_pnp_pose[:3, :3] = R_res
        lm_pnp_pose[:3, 3] = t.ravel()
     

        lm_pnp_poses.append(torch.Tensor(lm_pnp_pose[:3,:]))
        
    return torch.stack(lm_pnp_poses, dim=0)

def compute_err(pred_T, gt_T):
    gt_pose = gt_T.detach().cpu().numpy()
    pred_pose = pred_T.detach().cpu().numpy()
    R_acc = rel_R(pred_pose, gt_pose)
    t_acc = rel_t(pred_pose, gt_pose) 

    return R_acc, t_acc


def compute_err_batched(pred_Ts, gt_Ts):
    R_accs = []
    t_accs = []

    for idx, pred_T in enumerate(pred_Ts):
        # (3,4)
        gt_T = gt_Ts[idx, :, :].squeeze(0)
        # measure accuracy
        gt_pose = gt_T.detach().cpu().numpy()
        pred_pose = pred_T.detach().cpu().numpy()
        R_acc = rel_R(pred_pose, gt_pose)
        t_acc = rel_t(pred_pose, gt_pose) 
        R_accs.append(R_acc)
        t_accs.append(t_acc)
    return R_accs, t_accs



