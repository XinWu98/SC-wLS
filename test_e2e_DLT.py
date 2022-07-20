import argparse
import time
import numpy as np
import os
import os.path as osp
import configparser
import torch
import torchvision
import torch.optim
import torch.utils.data
from models.Dense_networks_e2e import DenseSC
from models.Dense_networks_e2e_GNN import DenseSC_GNN
from utils import save_checkpoint, save_path_formatter

from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from reloc_pipeline.utils_func import compute_pose_lm_pnp, compute_err, compute_err_batched
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import dsacstar


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
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', default=1e-4, type=float,
                    metavar='LR', help='learning rate for SC Net')
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
parser.add_argument('--resume_optim', action='store_true',
                    help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--model', choices=('DenseSC','DenseSC_GNN'),
                    default='DenseSC_GNN',
                    help='Model to train')
parser.add_argument('--config_file', type=str, help='configuration file')

parser.add_argument('--dataset', type=str, choices=('7Scenes', 'Cambridge'), default='7Scenes',
                    help='Dataset')
parser.add_argument('--uncertainty', action='store_true',
                    help='predict uncertianty of the scene coordinate in SC Net, not use.')

# E2E training
parser.add_argument('--e2e_training', action='store_true', default=0,
                    help='will train the SC Net end-to-end')

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

#OANet
# -----------------------------------------------------------------------------
# Network
parser.add_argument(
    "--net_depth", type=int, default=12, help=""
    "number of layers. Default: 12")
parser.add_argument(
    "--clusters", type=int, default=500, help=""
    "cluster number in OANet. Default: 500")
parser.add_argument(
    "--iter_num", type=int, default=0, help=""
    "iteration number in the iterative network. Default: 1")
parser.add_argument(
    "--net_channels", type=int, default=128, help=""
    "number of channels in a layer. Default: 128")
def str2bool(v):
    return v.lower() in ("true", "1")
parser.add_argument(
    "--share", type=str2bool, default=False, help=""
    "share the parameter in iterative network. Default: False")

# Loss
parser.add_argument('-Ea', '--Eig_alpha', type=float, help="weight of the trifocal loss", metavar='W',
                    default=10.0)
parser.add_argument('-Eb', '--Eig_beta', type=float, help='weight of the aux loss', metavar='W',
                    default=5e-3)
parser.add_argument(
    "--loss_classif", type=float, default=0.0, help=""
    "weight of the classification loss")
parser.add_argument(
    "--loss_essential", type=float, default=1.0, help=""
    "weight of the essential loss")
parser.add_argument(
    "--loss_essential_init_epo", type=int, default=20, help=""
    "initial epochs to run only the classification loss")
parser.add_argument(
    "--obj_geod_th", type=float, default=5, help=""
    "theshold for the good geodesic distance")

# Evaluation
parser.add_argument('--useLM', action='store_true',
                    help='will use LM Refinment')

# Dsacstar ransac
parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')
parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
	help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')
parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10,
	help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlierbeta', '-ib', type=float, default=0.5, 
	help='beta parameter of the soft inlier count; controls the softness of the sigmoid; lower means softer')

# Pretrained model
parser.add_argument('--pretrained-model', dest='pretrained_model', default=None, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--w-model', dest='w_model', default=None, metavar='PATH',
                    help='path to seperate weight model')

OUTPUT_SUBSAMPLE = 8

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

    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints/{}/test/{}/{}'.format(args.model, args.dataset, scene) / save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    output_writers = SummaryWriter(args.save_path / 'valid')

    settings = configparser.ConfigParser()
    with open(args.config_file, 'r') as f:
        settings.read_file(f)
    section = settings['hyperparameters']
   
    print("=> fetching scenes in '{}'".format(args.data))
    kwargs = dict(scene=args.scene, data_path=args.data, transform=None,
                  target_transform=None, seed=args.seed)

    if args.model.find('Dense') >= 0:
        if args.dataset == '7Scenes':
            from datasets.sevenscenes import SevenScenes
            val_set = SevenScenes(train=False, **kwargs)
        elif args.dataset == 'Cambridge':
            from datasets.cambridge import Cambridge
            val_set = Cambridge(train=False, **kwargs)
           
  
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

    args.batch_size = 1
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if args.epoch_size == 0:
        args.epoch_size = len(val_loader)

    # create model and init
    print("=> creating model")
    if args.model == 'DenseSC':
        SC_Net = DenseSC(args, scene_mean).to(device)
    elif args.model == 'DenseSC_GNN':
        SC_Net = DenseSC_GNN(args, scene_mean).to(device)

    # model init
    if args.pretrained_model:
        print("=> using pre-trained weights for SC net")
        loc_func = None if torch.cuda.is_available() else lambda storage, loc: storage
        checkpoint = torch.load(args.pretrained_model, map_location=loc_func)
        SC_Net.load_state_dict(checkpoint['state_dict'], strict=False)
       
        if args.w_model:
            print('weight model')
            Total_dict = torch.load(args.w_model, map_location=loc_func)['state_dict']
            new_dlt_dict = OrderedDict()
            for k,v in Total_dict.items():
                if k.find('wDLT')>=0 :
                    name = k[9:]  # remove `wDLT_net.` if SC.wDLT_Net load weights. 
                    new_dlt_dict[name] = v
            SC_Net.wDLT_net.load_state_dict(new_dlt_dict, strict=True)        
    else:
        SC_Net.init_weights()

    # for multi-gpu
    device_ids = [0]
    SC_Net = SC_Net.cuda()
    SC_Net = torch.nn.DataParallel(SC_Net, device_ids=device_ids)
    if isinstance(SC_Net, torch.nn.DataParallel):
        SC_Net = SC_Net.module
    
    logger = TermLogger(n_epochs=args.epochs, train_size=min(0, args.epoch_size),
                        valid_size=len(val_loader))
    logger.epoch_bar.start()

    args.epochs = 1
    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)
        print("val loader: ", len(val_loader))

        # evaluate on test set
        logger.reset_valid_bar()

        with torch.no_grad():
            validate_with_gt(args,
                            val_loader,
                            SC_Net,
                            epoch,
                            logger,
                            output_writers)

        torch.cuda.empty_cache()

    logger.epoch_bar.finish()


@torch.no_grad()
def validate_with_gt(args, val_loader, SC_Net, epoch, logger, output_writers=[]):
    global device, val_n_iter
    batch_time = AverageMeter()
      
    # switch to evaluate mode
    torch.cuda.empty_cache()
    SC_Net.eval()
    logger.valid_bar.update(0)

    error_t = []
    error_q = []
    
    end = time.time()
    for i, frames in enumerate(val_loader):
        Ts = frames['T'].to(device)
        Ks = frames['K'].to(device)
        # B*3*H*W(480*640)
        image = frames['image'].to(device)
        
        pred_X_3d, uncertainty, e_hat_post, prediction_grid_pad= SC_Net(None, image, Ts,
                                                                                Ks, epoch, test=True, ds=False)
      
        if not args.useLM:
            # compute errors of weighted LS pose 
            R_acc, t_acc = compute_err_batched(e_hat_post, Ts)

            error_t.extend(t_acc)
            error_q.extend(R_acc)       
        else:
            # LM refinement
            for idx, pred_X in enumerate(pred_X_3d):
                out_pose = torch.zeros((4, 4))
                dsacstar.forward_rgb_refine(
                    pred_X.unsqueeze(0).cpu(),
                    e_hat_post[idx:idx+1].cpu(),
                    out_pose,
                    args.hypotheses,
                    args.threshold,
                    Ks[idx][0, 0],
                    Ks[idx][0, 2],  # principal point assumed in image center
                    Ks[idx][1, 2],
                    args.inlieralpha,
                    args.maxpixelerror,
                    OUTPUT_SUBSAMPLE)
                R_acc, t_acc = compute_err(out_pose, Ts[idx].cpu())
                error_t.append(t_acc)
                error_q.append(R_acc)

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
    num_error = error_q.shape[0]
    if args.dataset == '7Scenes':
        pct5 = np.sum((error_t < 0.05) & (error_q < 5))
    else:
        if args.scene == 'greatcourt':
            pct5 = np.sum((error_t < 0.45) )
        elif args.scene == 'kingscollege':
            pct5 = np.sum((error_t < 0.38) )
        elif args.scene == 'oldhospital':
            pct5 = np.sum((error_t < 0.22) )
        elif args.scene == 'shopfacade':
            pct5 = np.sum((error_t < 0.15) )
        elif args.scene == 'stmaryschurch':
            pct5 = np.sum((error_t < 0.35) )
    print('median t,r error for DLT:', np.median(error_t), np.median(error_q))
    print('ptc5 for DLT:', pct5/num_error)
   
    return 


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.backends.cudnn.benchmark = True
    main()
