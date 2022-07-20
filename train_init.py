import argparse
import time
import csv

import numpy as np
import os
import torch
import torchvision
import torch.optim
import torch.utils.data
from models.Dense_networks import DenseSC

from utils import save_checkpoint, save_path_formatter
from optimizer import Optimizer
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

import os.path as osp
import configparser

from reloc_pipeline.utils_func import compute_pose_lm_pnp


parser = argparse.ArgumentParser(description='relocation test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--scene', type=str, help='Scene name')

parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
# @
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--glr', default=1e-4, type=float,
                    metavar='LR', help='learning rate for Weight Net')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')

parser.add_argument('--seed', default=7, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
# @
parser.add_argument('-f', '--training-output-freq', type=int,
                    help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)

parser.add_argument('--resume_optim', action='store_true',
                    help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--model', choices=('DenseSC'),
                    default='DenseSC',
                    help='Model to train')
parser.add_argument('--config_file', type=str, help='configuration file')

parser.add_argument('--dataset', type=str, choices=('7Scenes', 'Cambridge'), default='7Scenes',
                    help='Dataset')
parser.add_argument('--cal_training_pose', action='store_true',
                    help='will calculate pose in training and visualized')
parser.add_argument('--uncertainty', action='store_true',
                    help='will calculate uncertianty of the prediction')
parser.add_argument('--dsacstar', action='store_true', default=True,
                    help='will use the loss of dsacstar')

# Reprojecttion Loss
parser.add_argument('--inittolerance', '-itol', type=float, default=0.1,
                    help='switch to reprojection error optimization when predicted scene coordinate is within this tolerance threshold to the ground truth scene coordinate, in meters')

parser.add_argument('--mindepth', '-mind', type=float, default=0.1,
                    help='enforce  predicted scene coordinates to be this far in front of the camera plane, in meters')

parser.add_argument('--maxdepth', '-maxd', type=float, default=1000,
                    help='enforce that scene coordinates are at most this far in front of the camera plane, in meters')
parser.add_argument('--targetdepth', '-td', type=float, default=10,
	help='if ground truth scene coordinates are unknown, use a proxy scene coordinate on the pixel ray with this distance from the camera, in meters')

parser.add_argument('--softclamp', '-sc', type=float, default=100,
	help='robust square root loss after this threshold, in pixels')

parser.add_argument('--hardclamp', '-hc', type=float, default=1000,
	help='clamp loss with this threshold, in pixels')

# Dsacstar ransac
parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10,
	help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')
parser.add_argument('--weightrot', '-wr', type=float, default=1.0,
	help='weight of rotation part of pose loss')

parser.add_argument('--weighttrans', '-wt', type=float, default=100.0,
	help='weight of translation part of pose loss')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
	help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

# Pretrained model
parser.add_argument('--pretrained-model', dest='pretrained_model', default=None, metavar='PATH',
                    help='path to pre-trained Posenet model')

OUTPUT_SUBSAMPLE = 8
import cv2

# for PnP
repro_thres = 0.75
best_error = -1
n_iter = 0
val_n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    scene = args.scene

    assert args.batch_size == 1

    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints/{}_init/{}/{}'.format(args.model, args.dataset, scene) / save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
   
    training_writer = SummaryWriter(args.save_path)
    output_writers = SummaryWriter(args.save_path / 'valid')

    settings = configparser.ConfigParser()
    with open(args.config_file, 'r') as f:
        settings.read_file(f)
    section = settings['hyperparameters']
    max_grad_norm = section.getfloat('max_grad_norm', 0)
  
    print("=> fetching scenes in '{}'".format(args.data))
    kwargs = dict(scene=args.scene, data_path=args.data, transform=None,
                  target_transform=None, seed=args.seed)

    if args.model.find('Dense') >= 0:
        if args.dataset == '7Scenes':
            from datasets.sevenscenes import SevenScenes
            train_set = SevenScenes(train=True, augment=True, **kwargs)
            val_set = SevenScenes(train=False, **kwargs)
        elif args.dataset == 'Cambridge':
            from datasets.cambridge import Cambridge
            train_set = Cambridge(train=True,augment=True, **kwargs)
            val_set = Cambridge(train=False, **kwargs)
       

    print('{} samples found in train scenes{}'.format(len(train_set), args.scene))
    print('{} samples found in valid scenes{}'.format(len(val_set), args.scene))

    if args.dataset == '7Scenes':
        mean_file = osp.join('data', '7scenes_init', '7scenes_{}'.format(scene), 'coords_mean.txt')
        scene_mean = np.loadtxt(mean_file)
        scene_mean = torch.from_numpy(scene_mean)
    elif args.dataset == 'Cambridge':
        mean_file = osp.join('data', 'cambridge_init', '{}_coords_mean.txt'.format(scene))
        scene_mean = np.loadtxt(mean_file)
        scene_mean = torch.from_numpy(scene_mean)
    else:
        scene_mean = torch.zeros((3))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model and init
    print("=> creating model")
    if args.model == 'DenseSC':
        SC_Net = DenseSC(args, scene_mean).to(device)

    print('=> setting adam solver')

    # optimizer
    param_list = [{'params': filter(lambda p: p.requires_grad, SC_Net.parameters()), 'lr': args.lr}]
    kwargs = dict(betas=(args.momentum, args.beta))
    optimizer = Optimizer(params=param_list, base_lr=args.lr, method='adam',
                          weight_decay=args.weight_decay, **kwargs)
  
    resume_optim = args.resume_optim
    if args.pretrained_model:
        print("=> using pre-trained weights for SC net")
        loc_func = None if torch.cuda.is_available() else lambda storage, loc: storage
        checkpoint = torch.load(args.pretrained_model, map_location=loc_func)
        SC_Net.load_state_dict(checkpoint['state_dict'], strict=False)
        if resume_optim:
            optimizer.learner.load_state_dict(checkpoint['optim_state_dict'])
    else:
        SC_Net.init_weights()

    # for multi-gpu
    device_ids = [0]
    SC_Net = SC_Net.cuda()
    SC_Net = torch.nn.DataParallel(SC_Net, device_ids=device_ids)
    if isinstance(SC_Net, torch.nn.DataParallel):
        SC_Net = SC_Net.module

    with open(args.save_path / args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')        
        writer.writerow(['t_median', 'q_median', 'lr'])
  
    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size),
                        valid_size=len(val_loader))
    logger.epoch_bar.start()


    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        print("train loader: ", len(train_loader))
        print("val loader: ", len(val_loader))

        # train for one epoch
        if epoch > 0:
            logger.reset_train_bar()
            train_loss, med_t_train, med_q_train= train(args,
                                                        epoch,
                                                        train_loader,
                                                        SC_Net,
                                                        optimizer,
                                                        args.epoch_size,
                                                        logger,
                                                        training_writer,
                                                        max_grad_norm)
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()

        with torch.no_grad():
            t_median, q_median = validate_with_gt(args,
                                                val_loader,
                                                SC_Net,
                                                epoch,
                                                logger,
                                                output_writers)
                                                                                                        
        if args.cal_training_pose and epoch > 0:
            training_writer.add_scalar('med_t_train', med_t_train, epoch)
            training_writer.add_scalar('med_q_train', med_q_train, epoch)


        training_writer.add_scalar('median_t_val', t_median, epoch)
        training_writer.add_scalar('median_q_val', q_median, epoch)

        decisive_error = 50 * t_median + q_median
        assert decisive_error >= 0

        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = (decisive_error < best_error) or (epoch % 200 == 0)
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': SC_Net.state_dict(),
                'optim_state_dict': optimizer.learner.state_dict()
            },
            is_best, epoch)

        with open(args.save_path / args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(
                [t_median, q_median, optimizer.learner.param_groups[0]['lr']])
        torch.cuda.empty_cache()

    logger.epoch_bar.finish()


def train(args, epoch, train_loader, SC_Net, optimizer, epoch_size, logger, train_writer, 
          max_grad_norm=0.0):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    batch_size = args.batch_size
   
    # switch to train mode
    torch.cuda.empty_cache()
    SC_Net.train()

    end = time.time()
    logger.train_bar.update(0)

    error_t = []
    error_q = []
   
    for i, frames in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        Ts = frames['T'].to(device)
        Ks = frames['K'].to(device)       
        X_world = None
        image = frames['image'].to(device)
        
        # compute gradient and do Adam step
        optimizer.learner.zero_grad()

        loss_1, loss_2, loss_3, pred_X_3d, uncertainty, _, prediction_grid_pad = SC_Net(X_world, image, Ts,
                                                                                                Ks)

        loss = torch.mean(loss_1)           
        loss.backward()
      
        if log_losses:
            train_writer.add_scalar('dsac*_loss', loss_1.item(), n_iter)
           
        # record loss 
        losses.update(loss.item(), batch_size)

        # compute gradient and do Adam step
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(SC_Net.parameters(), max_grad_norm)
        optimizer.learner.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.cal_training_pose:
            R_acc, t_acc, _ = compute_pose_lm_pnp(Ts,
                                                  pred_X_3d,
                                                  Ks,
                                                  prediction_grid_pad.unsqueeze(0),
                                                  None,
                                                  uncertainty,
                                                  repro_thres=repro_thres)
            
            error_q.extend(R_acc)
            error_t.extend(t_acc)
            
        logger.train_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1
     
    if args.cal_training_pose:
        error_t = np.array(error_t)
        error_q = np.array(error_q)

        # if args.dataset == '7Scenes':
        #     pct5 = np.sum((error_t < 0.05) & (error_q < 5))
        # else:
        #     if args.scene == 'greatcourt':
        #         pct5 = np.sum((error_t < 0.45) )
        #     elif args.scene == 'kingscollege':
        #         pct5 = np.sum((error_t < 0.38) )
        #     elif args.scene == 'oldhospital':
        #         pct5 = np.sum((error_t < 0.22) )
        #     elif args.scene == 'shopfacade':
        #         pct5 = np.sum((error_t < 0.15) )
        #     elif args.scene == 'stmaryschurch':
        #         pct5 = np.sum((error_t < 0.35) )
       
        num_error = error_q.shape[0]
        # train_writer.add_scalar('pcts/5', pct5/num_error, epoch)      
       
        return losses.avg[0], np.median(error_t), np.median(error_q)
    else:
        return losses.avg[0],10,10


@torch.no_grad()
def validate_with_gt(args, val_loader, SC_Net, epoch, logger, output_writers=[]):
    global device, val_n_iter
    batch_time = AverageMeter()
    log_outputs = 1  # len(output_writers) > 0

    # switch to evaluate mode
    torch.cuda.empty_cache()
    SC_Net.eval()
    end = time.time()
    logger.valid_bar.update(0)

    error_t = []
    error_q = []
  
    for i, frames in enumerate(val_loader):
        Ts = frames['T'].to(device)
        Ks = frames['K'].to(device)
        image = frames['image'].to(device)

        loss_1, loss_2, loss_3, pred_X_3d, uncertainty, post_mask, prediction_grid_pad = SC_Net(None, image, Ts,
                                                                                                    Ks)
        
        if args.dsacstar:
            loss = torch.mean(loss_1)
           
        if log_outputs:  
            if args.dsacstar:
                output_writers.add_scalar('dsac_loss_val', loss_1.item(), val_n_iter)
           
        R_acc, t_acc, _ = compute_pose_lm_pnp(Ts,
                                              pred_X_3d,
                                              Ks,
                                              prediction_grid_pad.unsqueeze(0),
                                              None,
                                              uncertainty,
                                              repro_thres=repro_thres)

        error_q.extend(R_acc)
        error_t.extend(t_acc)
        
        val_n_iter += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                'valid: Time {}'.format(batch_time))
    logger.valid_bar.update(len(val_loader))

    error_t = np.array(error_t)
    error_q = np.array(error_q)

    # if args.dataset == '7Scenes':
    #     pct5 = np.sum((error_t < 0.05) & (error_q < 5))
    # else:
    #     if args.scene == 'greatcourt':
    #         pct5 = np.sum((error_t < 0.45) )
    #     elif args.scene == 'kingscollege':
    #         pct5 = np.sum((error_t < 0.38) )
    #     elif args.scene == 'oldhospital':
    #         pct5 = np.sum((error_t < 0.22) )
    #     elif args.scene == 'shopfacade':
    #         pct5 = np.sum((error_t < 0.15) )
    #     elif args.scene == 'stmaryschurch':
    #         pct5 = np.sum((error_t < 0.35) )
    
    num_error = error_q.shape[0]
    # output_writers.add_scalar('pcts_val/5', pct5/num_error, epoch)
 
    return np.median(error_t), np.median(error_q)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.backends.cudnn.benchmark = False
   
    main()
