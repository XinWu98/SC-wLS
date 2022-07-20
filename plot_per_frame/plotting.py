import os
import numpy as np
import matplotlib.pyplot as plt


TXTPATH = './results'
SAVEPATH_TRANS = './trans/figs_all'
SAVEPATH_ROT = './rot/figs_all'

# def tmplog(tmparray):
#     return np.log(tmparray)

def get_trans_rot(txtarray):
    assert len(txtarray.shape) == 2 and txtarray.shape[1] == 2
    seqlen = txtarray.shape[0]
    x_seq = np.arange(seqlen)
    trans_err = txtarray[:, 0].flatten()
    rot_err = txtarray[:, 1].flatten()
    return x_seq, trans_err, rot_err

def scatter_multi(x_seq, y_seqs, labels, colors, savepath, xlabel=None, ylabel=None, title=None, ymax=None):
    
    plt.subplot(121)
    
    plt.ylim(-0.025 * ymax, ymax * 1.025)
    
    for y_seq, one_label, one_color in zip(y_seqs, labels, colors):
        plt.scatter(x_seq, y_seq, s=5, c=one_color, alpha=0.7, label=one_label)
    
    plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Image number')
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title+'_clipped')
    
    plt.subplot(122)
    for y_seq, one_label, one_color in zip(y_seqs, labels, colors):
        plt.scatter(x_seq, y_seq, s=5, c=one_color, alpha=0.7, label=one_label)
    
    plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Image number')
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title+'_unclipped')
    
    plt.savefig(os.path.join(savepath, '%s.png' % title))
    # plt.show()
    plt.clf()

def main(seqname='pumpkin', ymax_trans=None, ymax_rot=None):
    
    methods = {
        'MS-Transformer': ('MST_%s.txt', 'skyblue'),
        'DSAC*': ('DSACSTAR_%s.txt', 'm'),
        'Ours(dlt+e2e)': ('DLTe2e_%s.txt', 'r'),
        'Ours(dlt+e2e+ref)': ('DLTe2e_ref_%s.txt', 'y'),
        'Ours(dlt+e2e+DSAC*)': ('DLTe2e_dsacstar_%s.txt', 'g')
    }
    
    x_seq = None
    t_errs = []
    r_errs = []
    labels = []
    colors = []
    for k, (v, c) in methods.items():
        try:
            txtarray = np.loadtxt(os.path.join(TXTPATH, v % seqname))
        except:
            continue
        else:
            x_seq, t_err, r_err = get_trans_rot(txtarray)
            t_errs.append(t_err)
            r_errs.append(r_err)
            labels.append(k)
            colors.append(c)
    
    os.makedirs(SAVEPATH_TRANS, exist_ok=True)
    os.makedirs(SAVEPATH_ROT, exist_ok=True)
    
    fig = plt.figure(figsize=(24,9))
    scatter_multi(x_seq, t_errs, labels, colors, SAVEPATH_TRANS, ylabel='Translation error(m)', title=seqname, ymax=ymax_trans)
    scatter_multi(x_seq, r_errs, labels, colors, SAVEPATH_ROT, ylabel='Rotation error(degree)', title=seqname, ymax=ymax_rot)


def plot_final():
    methods = {
        'MS-Transformer': ('MST_%s.txt', 'skyblue'),
        'DSAC*': ('DSACSTAR_%s.txt', 'm'),
        'Ours(dlt+e2e)': ('DLTe2e_%s.txt', 'r'),
        'Ours(dlt+e2e+ref)': ('DLTe2e_ref_%s.txt', 'y'),
        'Ours(dlt+e2e+DSAC*)': ('DLTe2e_dsacstar_%s.txt', 'g')
    }
    
    seqs = [
        ('office', 1.0, 40.0),
        ('kingscollege', 5.0, 30.0)
    ]
    
    fig = plt.figure(figsize=(12, 9))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.30)
    for inds, (seqname, ymax_trans, ymax_rot) in enumerate(seqs):
        
        x_seq = None
        t_errs = []
        r_errs = []
        labels = []
        colors = []
        for k, (v, c) in methods.items():
            txtarray = np.loadtxt(os.path.join(TXTPATH, v % seqname))
            x_seq, t_err, r_err = get_trans_rot(txtarray)
            t_errs.append(t_err)
            r_errs.append(r_err)
            labels.append(k)
            colors.append(c)
        
        plt.subplot(221 + inds*2)
        plt.ylim(-0.025 * ymax_trans, ymax_trans * 1.025)
        for y_seq, one_label, one_color in zip(t_errs, labels, colors):
            plt.scatter(x_seq, y_seq, s=2, c=one_color, alpha=0.7, label=one_label)
        plt.legend(loc='upper left')
        plt.xlabel('Image number')
        plt.ylabel('Translation error(m)')
        plt.title(seqname)
        
        plt.subplot(222 + inds*2)
        plt.ylim(-0.025 * ymax_rot, ymax_rot * 1.025)
        for y_seq, one_label, one_color in zip(r_errs, labels, colors):
            plt.scatter(x_seq, y_seq, s=2, c=one_color, alpha=0.7, label=one_label)
        plt.legend(loc='upper left')
        plt.xlabel('Image number')
        plt.ylabel('Rotation error(degree)')
        plt.title(seqname)
    plt.savefig('./result_s2.png', bbox_inches='tight')


if __name__ == '__main__':
    plot_final()
    
    # seqs = [
    #     # ('chess', 0.5, 20.0),
    #     # ('fire', 1.0, 90.0),
    #     # ('greatcourt', 125.0, 180.0),
    #     # ('heads', 0.5, 85.0),
    #     ('kingscollege', 5.0, 30.0),
    #     ('office', 1.0, 100.0),
    #     # ('oldhospital', 10.0, 18.0),
    #     # ('pumpkin', 2.0, 89.0),
    #     # ('redkitchen', 1.8, 80.0),
    #     # ('shopfacade', 2.0, 30.0),
    #     # ('stairs', 1.0, 30.0),
    #     # ('stmaryschurch', 10.0, 50.0),
    # ]
    # for (seq, ymax_trans, ymax_rot) in seqs:
    #     main(seqname=seq, ymax_trans=ymax_trans, ymax_rot=ymax_rot)
