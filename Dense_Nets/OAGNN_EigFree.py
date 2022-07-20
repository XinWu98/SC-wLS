import torch
import torch.nn as nn
from Dense_Nets.Super_GNN import AttentionalPropagation

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        # ResNet
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class squeeze(nn.Module):
    def __init__(self, dim1):
        nn.Module.__init__(self)
        self.dim1 = dim1
      
    def forward(self, x):
        return x.squeeze(self.dim1)
class unsqueeze(nn.Module):
    def __init__(self, dim1):
        nn.Module.__init__(self)
        self.dim1 = dim1
      
    def forward(self, x):
        return x.unsqueeze(self.dim1)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class AttentionalGNN_layer(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.layer = AttentionalPropagation(feature_dim, 4)
        
    def forward(self, desc0):
        delta0 = self.layer(desc0, desc0)
        return delta0

class OAFilter_GNN(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                squeeze(-1),
                AttentionalGNN_layer(channels),
                unsqueeze(-1),
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                squeeze(-1),
                AttentionalGNN_layer(out_channels),
                unsqueeze(-1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

# you can use this bottleneck block to prevent from overfiting when your dataset is small
class OAFilterBottleneck(nn.Module):
    def __init__(self, channels, points1, points2, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points1),
                nn.ReLU(),
                nn.Conv2d(points1, points2, kernel_size=1),
                nn.BatchNorm2d(points2),
                nn.ReLU(),
                nn.Conv2d(points2, points1, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x)# b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_up, x_down):
        #x_up: b*c*n*1
        #x_down: b*c*k*1
        embed = self.conv(x_up)# b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)# b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out



class OANBlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PointCN(channels))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PointCN(2*channels, channels))
        for _ in range(self.layer_num//2-1):
            self.l1_2.append(PointCN(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)


    def forward(self, data, xs, Ks):
        batch_size, num_pts = data.shape[0], data.shape[2]
        
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2(torch.cat([x1_1,x_up], dim=1))
       
        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        
        return logits, None


class OANBlock_GNN(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PointCN(channels))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter_GNN(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PointCN(2*channels, channels))
        for _ in range(self.layer_num//2-1):
            self.l1_2.append(PointCN(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)


    def forward(self, data, xs, Ks):
        batch_size, num_pts = data.shape[0], data.shape[2]
        
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2(torch.cat([x1_1,x_up], dim=1))
       
        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        
        return logits, None


class OANet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        print(config)
        self.iter_num = config['iter_num']
        depth_each_stage = config['net_depth']//(config['iter_num']+1)
        self.weights_init = OANBlock(config['net_channels'], 5, depth_each_stage, config['clusters'])
        # side information is weight (So far not include residual)
        self.weights_iter = [OANBlock(config['net_channels'], 6, depth_each_stage, config['clusters']) for _ in range(config['iter_num'])]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data, Ks):
        data_shp = data.size()
        input = data.permute(0,2,1).unsqueeze(-1)
      
        res_logits, res_e_hat = [], []
        logits, e_hat = self.weights_init(input, data, Ks)
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat = self.weights_iter[i](
                torch.cat([input, torch.relu(torch.tanh(logits)).view(data_shp[0],1,-1,1).detach()], dim=1), data, Ks)
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat  

class OAGNN(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        print(config)
        self.iter_num = config['iter_num']
        depth_each_stage = config['net_depth']//(config['iter_num']+1)
        self.weights_init = OANBlock_GNN(config['net_channels'], 5, depth_each_stage, config['clusters'])
        # side information is weight (So far not include residual)
        self.weights_iter = [OANBlock_GNN(config['net_channels'], 6, depth_each_stage, config['clusters']) for _ in range(config['iter_num'])]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data, Ks):
        data_shp = data.size()
        input = data.permute(0,2,1).unsqueeze(-1)
      
        res_logits, res_e_hat = [], []
        logits, e_hat = self.weights_init(input, data, Ks)
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat = self.weights_iter[i](
                torch.cat([input, torch.relu(torch.tanh(logits)).view(data_shp[0],1,-1,1).detach()], dim=1), data, Ks)
           
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat  


        
def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
   
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.sigmoid(logits)
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4),  b*4*n, (u,v,u',v')
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm, (b,n,9)
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)

    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X

    XwX = torch.matmul(X.permute(0, 2, 1), wX)    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


def weighted_6points(x_in, logits, Ks, is_train):
    # x_in: batch * N * 5
    x_shp = x_in.shape
    # Turn into weights for each sample, (b,n)
    # weights = torch.relu(torch.tanh(logits))
    weights = torch.sigmoid(logits)

    # Make input data (num_img_pair x num_corr x 4), b*5*n, (u,v,x,y,z)
    xx = x_in.view(x_shp[0], x_shp[1], 5).permute(0, 2, 1)
    fx, fy, cx, cy = Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]

    # Create the matrix to be used for the eight-point algorithm, (b,n,12)
    # X_up concat x_down
    cx_u = (cx.unsqueeze(1) - xx[:, 0])
    cy_v = (cy.unsqueeze(1) - xx[:, 1])
    fx = fx.unsqueeze(-1)
    fy = fy.unsqueeze(-1)

    X_up = torch.stack([
        fx * xx[:, 2], fx * xx[:, 3], fx * xx[:, 4], fx * torch.ones_like(xx[:, 2]),
        torch.zeros_like(xx[:, 2]), torch.zeros_like(xx[:, 2]), torch.zeros_like(xx[:, 2]), torch.zeros_like(xx[:, 2]),
        xx[:, 2] * cx_u, xx[:, 3] * cx_u, xx[:, 4] * cx_u, cx_u
    ], dim=1).permute(0, 2, 1)
    X_bottom = torch.stack([
        torch.zeros_like(xx[:, 2]), torch.zeros_like(xx[:, 2]), torch.zeros_like(xx[:, 2]), torch.zeros_like(xx[:, 2]),
        fy * xx[:, 2], fy * xx[:, 3], fy * xx[:, 4], fy * torch.ones_like(xx[:, 2]),
        xx[:, 2] * cy_v, xx[:, 3] * cy_v, xx[:, 4] * cy_v, cy_v
    ], dim=1).permute(0, 2, 1)
 
    X = torch.cat([X_up, X_bottom], dim=1)


    w_cat = torch.cat([weights, weights], dim=-1)
    wX = w_cat.unsqueeze(-1) * X

    XwX = torch.matmul(X.permute(0, 2, 1), wX)
   
    v = batch_symeig(XwX)   
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 3, 4))
   
    R_hat = e_hat[:,:,:3]
    t_hat = e_hat[:,:,3]

    U,D,V = torch.svd(R_hat)
    
    c = torch.reciprocal(D.sum(dim=-1)/3.0).unsqueeze(-1)
    _, indice = torch.max(weights, dim=-1)
   
    #(b,3)
    Point_verify = xx[torch.arange(x_shp[0]), 2:, indice]
    ones = torch.ones([x_shp[0], 1]).to(device)
    # (b,1,4)
    Point_verify = torch.cat([Point_verify, ones], dim=-1).unsqueeze(1)
   
    criterion = c * (torch.matmul(Point_verify, e_hat[:,-1].view(x_shp[0], 4, 1)).squeeze(-1))
  
    inv_c = -1*c
    c = torch.where(criterion > 0, c, inv_c)
 
    R_hat = torch.matmul(U, V.permute(0,2,1)) * torch.sign(c.unsqueeze(-1))
    t_hat = c * t_hat
   
    #(b,3,4)
    e_hat = torch.cat([R_hat, t_hat.unsqueeze(-1)], dim=-1)
  
    return e_hat

