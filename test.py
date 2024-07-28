import os
from os import path as osp
import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import torch.distributions.multivariate_normal as torchdist
from utils.metrics import * 
from models.TrajectoryModel import TrajectoryModel
from data_loader import inDDatasetGraph, TrajectoryDataset
from models.TrajectoryModel import *
import copy
from tqdm import tqdm
import argparse
from collections import OrderedDict

import seaborn as sns


parser = argparse.ArgumentParser(description="Parameter Settings for Training")

parser.add_argument('--pretrained_file', default="model_9.pt",
                help="Path to pretrained file which wants to be validated")
# --- Input ---
# dataset options
parser.add_argument('--dataset_dir', default="./datasets/",
                help="Path to directory that contains the dataset csv files.", type=str)
parser.add_argument('--dataset', default="inD-dataset-v1.0",
                help="Name of the dataset. Needed to apply dataset specific visualization adjustments.",
                type=str)
parser.add_argument('--sdd_loc', default="",
                help="Location of SDD", type=str)
parser.add_argument('--pretrained_dir', default="./pretrained",
                help="Path to directory that contains the pretrained model.", type=str)
parser.add_argument('--recording', default="26",
                help="Name of the recording given by a number with a leading zero.", type=str)

# training options
parser.add_argument('--seed', default=42,
                help="seed number", type=int)
parser.add_argument('--epochs', default="10",
                help="Epochs for train, validation, and test suite", type=int)
parser.add_argument('--train_num', default="24",
                help="Number of training dataset the model splits", type=int)
parser.add_argument('--test_num', default="8",
                help="Number of testing dataset the model splits", type=int)
parser.add_argument('--bs', default="4",
                help="Batch size", type=int)
parser.add_argument('--optimizer', default="Adam",
                help="Name of the optimizer we use for train", type=str)
parser.add_argument('--model_name', default="social_stgcnn",
                help="Model name", type=str)
parser.add_argument('--is_pretrained', action="store_true", default=False,
                help="Use pretrained model")
parser.add_argument('--pretrained_epoch', default=None,
                help="Name of pretrained model", type=int)
parser.add_argument('--num_timesteps_in', default="8",
                help="Number of timesteps for input", type=int)
parser.add_argument('--num_timesteps_out', default="12",
                help="Number of timesteps for output", type=int)
parser.add_argument('--rn_num_timesteps_in', default="10",
                help="Number of timesteps for road network input", type=int)
parser.add_argument('--rn_num_timesteps_out', default="10",
                help="Number of timesteps for road network output", type=int)
parser.add_argument('--is_horizontal_pred', default=False,
                help="If the model is trained time horizontaly, it is true", type=bool)
parser.add_argument('--horizon', default="4",
                help="Number of horizon")
parser.add_argument('--grid', default="4",
                help="Number of grid on one side")
parser.add_argument('--agg_frame', default="20",
                help="Aggregated number of frames")
parser.add_argument('--is_rn', action="store_true", default=False,
                help="If road network is taken in the learning phase")
parser.add_argument('--is_preprocessed', action="store_true", default=False,
                help="If preprocessed file exists")
parser.add_argument('--tr', '--train_ratio', default=0.8, type=float,
                help="Train ratio")
parser.add_argument('--fusion', default=None, type=str,
                help="Feature fusion method")
parser.add_argument('--is_visualize', action="store_true", default=False,
                help="If you want to visualize the results")
parser.add_argument('--temporal', default=None, type=str,
                help="Temporal model name")
parser.add_argument('--skip', default=12, type=int,
                help="Frame adjustable parameter. 12 is 2.5 FPS for SDD.")
parser.add_argument('--is_normalize', action="store_true", default=False,
                help="If you want to normalize")
parser.add_argument('--uid', default=0, type=int,
                help="Unique ID")

opt = parser.parse_args()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(opt.seed)

paths = ['./checkpoints']
KSTEPS = 20

print("*"*50)
print('Number of samples:', KSTEPS)
print("*"*50)

model_path = osp.join(opt.pretrained_dir, opt.model_name, opt.pretrained_file) # '/val_best.pth'

# Data prep     
obs_seq_len = opt.num_timesteps_in
pred_seq_len = opt.num_timesteps_out


def load_data(opt):
    dataset_dir = osp.join(opt.dataset_dir, opt.dataset)

    print(obs_seq_len, pred_seq_len)

    if opt.dataset == 'inD-dataset-v1.0':
        test_dataset = inDDatasetGraph(
            dataset_dir,
            num_timesteps_in=opt.num_timesteps_in,
            num_timesteps_out=opt.num_timesteps_out,
            train_num=opt.train_num,
            test_num=opt.test_num,
            skip=10,
            grid=opt.grid,
            is_preprocessed=opt.is_preprocessed,
            dataset=opt.dataset,
            sdd_loc=opt.sdd_loc)

        test_loader = DataLoader(
                test_dataset,
                batch_size=1, # This is irrelative to the args batch size parameter
                shuffle=False,
                num_workers=1)
    else:
        if opt.is_rn:
            out_list = [1, 4, 8]
        else:
            out_list = [opt.num_timesteps_out]

        test_rn_loader_list = []

        for i in range(len(out_list)):
            test_dataset = TrajectoryDataset(
                    dataset_dir,
                    sdd_loc=opt.sdd_loc,
                    in_channels=opt.num_timesteps_in,
                    out_channels=opt.num_timesteps_out,
                    rn_out_channels=out_list[i],
                    agg_frame=opt.agg_frame,
                    skip=opt.skip,
                    grid=opt.grid,
                    norm_lap_matr=True,
                    is_preprocessed=opt.is_preprocessed,
                    dataset=opt.dataset,
                    dataset_iter=i,
                    train_mode='test',
                    is_rn=opt.is_rn,
                    is_normalize=opt.is_normalize)

            if opt.is_rn:
                test_rn_dataset = test_dataset.get(opt.rn_num_timesteps_in, out_list[i])
                test_rn_loader_list.append(test_rn_dataset)

            if i == 0:
                test_loader = DataLoader(
                        test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0)

    if opt.is_rn:
        return test_loader, test_rn_loader_list
    else:
        return test_loader


def world2image(traj_w, H_inv):
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T  
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)  
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2]) 
    return traj_uvz[:, :2].astype(int)    

def get_closer_index(key, query):
    diff = np.linalg.norm(query - key, axis=-2).sum(-1)
    argmin = np.argmin(diff,axis=0)
    # print(diff[argmin], argmin, "\n",key,"\n",query[argmin])
    return argmin

def obtain_limits(model_data, normal_xy_ratio=True):
    x_min = y_min = np.inf
    x_max = y_max = -np.inf
    for trajs in model_data.values():
        sub_x_min = trajs[..., 0, :].min()
        sub_x_max = trajs[..., 0, :].max()
        sub_y_min = trajs[..., 1, :].min()
        sub_y_max = trajs[..., 1, :].max()
        x_min = min(x_min, sub_x_min)
        x_max = max(x_max, sub_x_max)
        y_min = min(y_min, sub_y_min)
        y_max = max(y_max, sub_y_max)
    x_len = x_max - x_min
    y_len = y_max - y_min
    
    if normal_xy_ratio:
        if x_len/y_len > 3.5/4.5:  # the x/y ratio of plot region in the figure is about 3.5/4.5. 
            # y_len is too small
            y_center = (y_max + y_min) / 2
            y_len = x_len / 3.5 * 4.5
            y_min = y_center - y_len / 2
            y_max = y_center + y_len / 2
        else:
            # x_len is too small
            x_center = (x_max + x_min) / 2
            x_len = y_len / 4.5 * 3.5
            x_min = x_center - x_len / 2
            x_max = x_center + x_len / 2
        
    return [x_min, x_max], [y_min, y_max]

def visualize(raw_data_dic_, out_list, gt_list):
    '''
    Visualize trajectories (Prediction, Target, Observation)
    '''

    seq = [3]
    color_codes = ['b', 'y', 'orange', 'm', 'k', 'c']
    print(f'Sequence: {len(raw_data_dic_)}')
    print(f'Modes: {raw_data_dic_[0].keys()}')

    # Image can be eth or eth2
    img = plt.imread('figures/{}.jpg'.format(opt.dataset))
    model_obs, model_trgt, model_pred = [], [], []

    for seq in range(len(raw_data_dic_)):
        model_obs.append(raw_data_dic_[seq]['obs'])
        model_trgt.append(raw_data_dic_[seq]['trgt'])
        _pred = []
        for i in range(len(raw_data_dic_[seq]['pred'])):
            _pred.append(np.stack(raw_data_dic_[seq]['pred'][i]))
        model_pred.append(_pred)

    _model = {}
    _model['obs'] = np.concatenate(model_obs, axis=1).squeeze()
    _model['trgt'] = np.concatenate(model_trgt, axis=1).squeeze()

    _model['trgt'] = _model['trgt'] - _model['obs'][..., 0:1]
    full_traj = np.concatenate((_model['obs'], _model['trgt']), axis=-1)
    _model['trgt_ref'] = (full_traj[..., 1:] - full_traj[..., :-1])[..., -opt.num_timesteps_out:]

    _model['pred'] = np.transpose(
        np.concatenate(model_pred, axis=2).squeeze(),
        axes=(2, 0, 3, 1)) - _model['obs'][:, None, :, 0:1]
    _model['obs'] = _model['obs'] - _model['obs'][..., 0:1]

    print(len(_model['obs']))

    collect = []
    for key in range(len(_model['obs'])):
        collect.append(OrderedDict())
        key_value = _model['obs'][key]

        key_model = get_closer_index(key_value, _model['obs'])
        collect[-1]['Observation'] = _model['obs'][key]
        collect[-1]['GroundTruth'] = _model['trgt'][key]
        collect[-1]['Prediction'] = _model['pred'][key_model]

    cmap = ['g', 'y', 'r', 'k', 'r']
    indices = {}
    # Indices: 0, 1, 2, 3, 4, 50, 57, 58, 59, 65
    # Used for visualization
    # indices['eth'] = [65] # indices for the visualization
    # indices['eth'] = [i for i in range(57, 60)] # indices for the visualization
    dset = opt.dataset

    out_, gt_, gt2_ = [], [], []
    from scipy.ndimage import gaussian_filter

    _out_list, _gt_list = [], [] # 36 grids
    if opt.dataset == 'eth':
        ts = 68
    elif opt.dataset == 'univ':
        ts = 40
    elif opt.dataset == 'sdd':
        ts = 55
        # ts = 123

    indices['sdd'] = [ts]
    # Used for RN visualization
    indices['eth'] = [ts]
    indices['univ'] = [ts]

    # Iterate through each trajectory
    _show = 1
    for j, key in enumerate(indices[dset]):

        if dset in ['eth', 'zara1']:
            for model in collect[key].keys():
                collect[key][model] = np.flip(collect[key][model], axis=-2)

        for i in range(_show):
            if i == 0:
                plt.plot(collect[key]['Observation'][0],
                        collect[key]['Observation'][1],
                        '--o',
                        label='Observation',
                        color='b')
                plt.plot(collect[key]['GroundTruth'][0],
                        collect[key]['GroundTruth'][1],
                        '--x',
                        label='GroundTruth',
                        color='orange')

        # TODO: Visualize whole model prediction in one plot
        x_limit, y_limit = obtain_limits(collect[key])
        print(x_limit, y_limit)

        # if opt.dataset == 'eth':
        #     x_limit, y_limit = (-7.8025931517283125, 7.029090325037639), (0.0, 19.069307)
        # elif opt.dataset == 'univ':
        #     x_limit, y_limit = (-7.5070825, 0.0), (-3.064489568982806, 6.58747359684535)

        if opt.is_rn:
            for site_num in range(36):
                out_ = out_list[2][ts][site_num][0]
                g2_ = gt_list[2][ts][site_num][0]
                _out_list.append(out_)
                _gt_list.append(gt2_)
            _out_reshaped = np.array(_out_list).reshape(6, 6)
            data_smoothed = gaussian_filter(_out_reshaped, sigma=1)

        sns.kdeplot(x=collect[key]['Prediction'][:, 0].reshape(-1),
                    y=collect[key]['Prediction'][:, 1].reshape(-1),
                    fill=True,
                    thresh=0.15,
                    color='g',
                    alpha=0.8,
                    label=opt.model_name)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.axis('off')
        plt.xlim(x_limit) 
        plt.ylim(y_limit)
        plt.imshow(img, extent=[x_limit[0], x_limit[1], y_limit[0], y_limit[1]])
        plt.imshow(data_smoothed, cmap='coolwarm', interpolation='nearest', alpha=0.6, extent=[x_limit[0], x_limit[1], y_limit[0], y_limit[1]], aspect='auto')

        plt.tight_layout()
        plt.savefig(osp.join('../', 'figs', '{}_{}_in{}_out_{}_pretrained{}_frames{}.png'.format(opt.model_name, opt.dataset, opt.num_timesteps_in, opt.num_timesteps_out, opt.pretrained_epoch, key)),
                dpi=300,
                bbox_inches='tight')
        plt.show()


    plt.close()

def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)

@torch.no_grad()
def test(test_loader, test_rn_dataset=None, KSTEPS=20):
    model.eval()
    ade_bigls = [[] for _ in range(4)]
    rn_pred_list = [[] for _ in range(3)]
    gt_list = [[] for _ in range(3)]
    fde_bigls = []
    coll_bigls = []
    coll_step_bigls = []
    raw_data_dict = {}
    traj_pred = None
    loss = 0
    rn_loss = 0
    rn_loss1, rn_loss2, rn_loss3 = 0, 0, 0
    cnt = 0
    rn_loss_list = [[] for _ in range(3)]

    mabs_loss = []
    kde_loss = []
    m_collect = []
    eig_collect = []


    if opt.is_rn:
        if len(test_rn_dataset) == 1:
            pbar = tqdm(enumerate(zip(test_loader, test_rn_dataset[0])))
        else:
            pbar = tqdm(enumerate(zip(test_loader, test_rn_dataset[0], test_rn_dataset[1], test_rn_dataset[2])))
    else:
        pbar = tqdm(enumerate(test_loader))

    for i, batch in pbar: 
        pbar.update(1)
        cnt += 1
        loc_data = []

        if opt.is_rn:
            ### Train Local Peds Trajectory
            # Get data
            for tensor in batch[0]:
                if not isinstance(tensor, list):
                    loc_data.append(tensor.to(device))
                else:
                    loc_data.append(tensor)
            # loc_data = [tensor.to(device) for tensor in batch[0][:-3]]
            # Get data
            x = [batch[i].x.to(device) for i in range(1, len(batch))]
            y = [batch[i].y.to(device) for i in range(1, len(batch))]
        else:
            # loc_data = [tensor.to(device) for tensor in batch]
            for tensor in batch:
                if not isinstance(tensor, list):
                    loc_data.append(tensor.to(device))
                else:
                    loc_data.append(tensor)

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, min_rel_pred, max_rel_pred, ped_list = loc_data

        num_of_objs = obs_traj_rel.shape[1]
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        if opt.is_rn:

            rn_edge_index = [batch[i].edge_index.to(device) for i in range(1, len(batch))]
            rn_edge_attr = [batch[i].edge_attr.to(device) for i in range(1, len(batch))]

            ### Model in one piece
            rn_pred, traj_pred = model(V_obs_tmp, A_obs, x, rn_edge_index, rn_edge_attr, ped_list=ped_list)

            rn_loss1 += torch.mean((rn_pred[0] - y[0])**2).cpu()
            rn_loss2 += torch.mean((rn_pred[1] - y[1])**2).cpu()
            rn_loss3 += torch.mean((rn_pred[2] - y[2])**2).cpu()
            for j in range(len(rn_pred)):
                rn_loss += torch.mean((rn_pred[j] - y[j])**2).cpu()
                rn_pred_list[j].append(rn_pred[j].cpu().detach().numpy())
                gt_list[j].append(y[j].cpu().detach().numpy())
                rn_loss_list[j].append(torch.mean((rn_pred[j] - y[j])**2).cpu())
        else:
            if (V_obs_tmp.size(-1) == 0):
                break
            rn_pred, traj_pred = model(V_obs_tmp, A_obs, ped_list=ped_list)


        if opt.model_name != "social_lstm":
            traj_pred = traj_pred.permute(0, 2, 3, 1)

        traj_loss = 0
        loss += traj_loss

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        traj_pred = traj_pred.squeeze() #.float()
        num_of_objs = obs_traj_rel.shape[1]

        if opt.model_name != "social_implicit":
            sx = torch.exp(traj_pred[:, :, 2]) #sx
            sy = torch.exp(traj_pred[:, :, 3]) #sy
            corr = torch.tanh(traj_pred[:, :, 4]) #corr
            
            cov = torch.zeros(traj_pred.shape[0], traj_pred.shape[1], 2, 2).cuda()
            cov[:, :, 0, 0] = sx * sx
            cov[:, :, 0, 1] = corr * sx * sy
            cov[:, :, 1, 0] = corr * sx * sy
            cov[:, :, 1, 1] = sy * sy
            mean = traj_pred[:, :, 0:2]
            
            mvnormal = torchdist.MultivariateNormal(mean, cov)

        # sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())

        V_x_rel_to_abs = seq_to_nodes(obs_traj.data.cpu().numpy().copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze(),
                                                    V_x[-1, :, :].copy())

        
        raw_data_dict[i] = {}
        raw_data_dict[i]['obs'] = copy.deepcopy(obs_traj.data.cpu().numpy().copy())
        raw_data_dict[i]['trgt'] = copy.deepcopy(pred_traj_gt.data.cpu().numpy().copy())
        raw_data_dict[i]['pred'] = []

        ade_ls = [[[] for _ in range(num_of_objs)] for _ in range(4)]
        fde_ls = [[] for _ in range(num_of_objs)]
        coll_ls = {}
        coll_step_ls = {}
        for n in range(num_of_objs):
            coll_ls[n] = []
            coll_step_ls[n] = []


        b_samples = []
        for k in range(KSTEPS):

            if opt.model_name == "social_implicit":
                V_pred = traj_pred[k:k+1, ...].squeeze()
            else:
                V_pred = mvnormal.sample()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1, :, :].copy())
            raw_data_dict[i]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
            # b_samples.append(V_pred_rel_to_abs[:, :, None, :].copy())

            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n+1, :])
                target.append(V_y_rel_to_abs[:, n:n+1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n+1, :])
                number_of.append(1)

                adeAll, ade1, ade2, ade3 = ade(pred, target, number_of)
                # ade_ls[n].append(ade(pred, target, number_of))
                ade_ls[0][n].append(adeAll)
                ade_ls[1][n].append(ade1)
                ade_ls[2][n].append(ade2)
                ade_ls[3][n].append(ade3)
                fde_ls[n].append(fde(pred, target, number_of))

                predicted_traj = V_pred_rel_to_abs[:, n, :].copy()
                predicted_trajs_all = V_pred_rel_to_abs[:, :, :].copy().transpose(1, 0, 2)

                col_mask = compute_col(predicted_traj, predicted_trajs_all).astype(np.float64)  # [56]
                if col_mask.sum():
                    coll_ls[n].append(1)
                else:
                    coll_ls[n].append(0)

                coll_step_ls[n].append(col_mask)

        for key, coll_step_data in zip(coll_step_ls.keys(), coll_step_ls.values()):
            coll_step_ls[key] = np.stack(coll_step_data, axis=0)  # [X, 56]

        # abs_samples = np.concatenate(b_samples, axis=2)  #ab samples in (12,3,100,2) gt in (12,3,2)

        # m, nan_list, n_u, m_c, eig = calc_amd_amv(V_y_rel_to_abs.copy(),
        #                                           abs_samples.copy())
        # mabs_loss.append(m) # m
        # eig_collect.append(eig)
        # _kde = kde_lossf(V_y_rel_to_abs.copy(), abs_samples.copy())
        # kde_loss.append(_kde)
        # m_collect.extend(m_c) # m_c
        
        for n in range(num_of_objs):
            # ade_bigls.append(min(ade_ls[n]))
            ade_bigls[0].append(min(ade_ls[0][n]))
            ade_bigls[1].append(min(ade_ls[1][n]))
            ade_bigls[2].append(min(ade_ls[2][n]))
            ade_bigls[3].append(min(ade_ls[3][n]))
            fde_bigls.append(min(fde_ls[n]))
            coll_bigls.append(sum(coll_ls[n]) / len(coll_ls[n]))

        coll_step_bigls.append(np.concatenate([ls for ls in coll_step_ls.values()], axis=0))  # [X, 56]

    coll_raw_ = np.concatenate(coll_step_bigls, axis=0)  # [X, 56]
    coll_step_ = np.mean(coll_raw_, axis=0)  # [56]
    coll_step_ = coll_step_[:-1].reshape(-1, 5).mean(axis=1)  # [11]
    coll_cumulative_ = np.asarray([np.mean(coll_raw_[:, :i * 5 + 6].max(axis=1)) for i in range(11)])  # [11]

    coll_ = sum(coll_bigls) / len(coll_bigls)

    print(f'Loss: {traj_loss / cnt}')

    # ade1_ = sum(ade_bigls) / len(ade_bigls)
    ade1_ = sum(ade_bigls[0]) / len(ade_bigls[0])
    ade2_ = sum(ade_bigls[1]) / len(ade_bigls[1])
    ade3_ = sum(ade_bigls[2]) / len(ade_bigls[2])
    ade4_ = sum(ade_bigls[3]) / len(ade_bigls[3])
    fde_ = sum(fde_bigls) / len(fde_bigls)
    rn_loss = rn_loss / cnt
    rn_loss1 = rn_loss1 / cnt
    rn_loss2 = rn_loss2 / cnt
    rn_loss3 = rn_loss3 / cnt

    # return ade1_, fde_, raw_data_dict
    return ade1_, ade2_, ade3_, ade4_, fde_, coll_, coll_cumulative_, raw_data_dict, rn_pred_list, gt_list, rn_loss_list #, #sum(kde_loss) / len(kde_loss), sum(mabs_loss) / len(mabs_loss), sum(eig_collect) / len(eig_collect), raw_data_dict

def rn_visualize(out_list, gt_list):
    out_, gt_, gt2_ = [], [], []
    # print(len(out_list))
    # print(out_list[0].shape)
    # print(gt_list[0].shape)
    from scipy.ndimage import gaussian_filter

    site_num = 15
    timestep = 0 # out timestep??

    _out_list, _gt_list = [], [] # 36 grids
    if opt.dataset == 'eth':
        ts = 58
    elif opt.dataset == 'univ':
        ts = 4

    # for i, out in enumerate(out_list):
    for site_num in range(36):
        out_ = out_list[ts][site_num][0]
        g2_ = gt_list[ts][site_num][0]
        _out_list.append(out_)
        _gt_list.append(gt2_)
    _out_reshaped = np.array(_out_list).reshape(6, 6)
    data_smoothed = gaussian_filter(_out_reshaped, sigma=1)

    # print(_out_reshaped)

    plt.figure(figsize=(10, 10))
    sns.heatmap(data_smoothed, annot=False, fmt=".2f", cmap='coolwarm', cbar=False)
    plt.imshow(data_smoothed, cmap='coolwarm', interpolation='nearest', alpha=0.6, extent=[], aspect='auto')
    plt.show()



if __name__ == '__main__':


    ade_ls = [] 
    ade2_ls = [] 
    ade3_ls = [] 
    ade4_ls = [] 
    fde_ls = [] 
    kde_ls = []
    amd_ls = []
    eig_ls = []
    coll_ls = []

    global out_list

    h = None
    out_list = [1, 4, 8]
    num_nodes = 0

    if opt.is_rn:
        test_loader, test_rn_dataset = load_data(opt)
    else:
        test_loader = load_data(opt)

    ### Model saving directory
    print("===> Save model to %s" % opt.pretrained_dir)

    os.makedirs(opt.pretrained_dir, exist_ok=True)

    global model, model_rn
    if opt.is_rn:
        print("===== Initializing model for time horizon =====")
        num_nodes = len(next(iter(test_rn_dataset[0])).x)
        print(num_nodes)

    ### Model setup
    print("===== Initializing model for trajectory prediction =====")
    model_rn = None
    model_loc = None
    # model_rn = RNTransformer(node_features=7, num_nodes=num_nodes, periods=opt.num_timesteps_in, output_dim_list=out_list, device=device).to(device)
    # model_path = osp.join(opt.pretrained_dir, 'road_network', 'eth', '{}_model_grid{}_outlist{}_{}_{}_epoch{}.pt'.format(11, opt.grid, out_list[0], out_list[1], out_list[2], 50))
    # model_rn.load_state_dict(torch.load(model_path))
    model = TrajectoryModel(in_channels=opt.num_timesteps_in, out_channels=opt.num_timesteps_out,
                                num_nodes=num_nodes, out_list=out_list, periods=opt.num_timesteps_in, 
                            depth=1, mlp_dim=128, device=device, is_rn=opt.is_rn, model_name=opt.model_name, model_loc=model_loc, model_rn=model_rn).to(device)

    # Check if pretrained model exists
    if opt.pretrained_epoch == None:
        if opt.is_rn:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_rn_best.pt'.format(opt.uid)), map_location='cuda:0'))
        else:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_best.pt'.format(opt.uid))))
    else:
        if opt.dataset == 'sdd':
            d_name = 'sdd'
        else:
            d_name = 'eth'
        if opt.is_rn:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_grid{}_epoch{}.pt'.format(opt.uid, opt.grid, opt.pretrained_epoch)), map_location='cuda:0'))
        else:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_{}.pt'.format(opt.uid, opt.pretrained_epoch))))

    total_param = 0
    for param_tensor in model.state_dict():
        # print("{}, {}".format(param_tensor, model.state_dict()[param_tensor].size()))
        total_param += np.prod(model.state_dict()[param_tensor].size())
        # print(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    stats = osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, 'constant_metrics.pkl')
    with open(stats, 'rb') as f:
        cm = pickle.load(f)
    print("Stats:", cm)

    print(f'Pretrained Epoch: {opt.pretrained_epoch}')

    ade_ = 999999
    fde_ = 999999
    ade2_, ade3_, ade4_  = 999999, 999999, 999999

    print("Testing ....")
    test_num = 5
    for _ in range(test_num):
        if opt.is_rn:
            # ad, fd, kd, md, eg, raw_data_dic_= test(test_loader, test_rn_dataset)
            # ad, fd, raw_data_dic_= test(test_loader, test_rn_dataset)
            ad, ad2, ad3, ad4, fd, coll_, coll_cum, raw_data_dic_, rn_pred_list, gt_list, rn_loss_list = test(test_loader, test_rn_dataset)
        else:
            # ad, fd, kd, md, eg, raw_data_dic_= test(test_loader)
            # ad, fd, raw_data_dic_= test(test_loader)
            ad, ad2, ad3, ad4, fd, coll_, coll_cum, raw_data_dic_, rn_pred_list, gt_list, rn_loss_list = test(test_loader)
        ade_ = min(ade_, ad)
        fde_ = min(fde_, fd)
        ade2_ = min(ade2_, ad2)
        ade3_ = min(ade3_, ad3)
        ade4_ = min(ade4_, ad4)
        ade_ls.append(ade_)
        ade2_ls.append(ade2_)
        ade3_ls.append(ade3_)
        ade4_ls.append(ade4_)
        fde_ls.append(fde_)
        coll_ls.append(coll_)
        # kde_ls.append(kd)
        # amd_ls.append(md)
        # eig_ls.append(eg)
        print("ADE:", ade_, " FDE:", fde_, " ADE1:", ade2_, " ADE2:", ade3_, " ADE4:", ade4_, " Coll:", coll_)#, " RN Loss1:", rn_loss_list[0], " RN Loss2:", rn_loss_list[1], " RN Loss3:", rn_loss_list[2]) #, "AMD:", md, "KDE:", kd, "AMV:", eg)

    print("*"*50)

    print("Avg ADE:", sum(ade_ls) / test_num)
    print("ADE First Period: {} Second Period: {} Third Period: {}".format(sum(ade2_ls)/test_num, sum(ade3_ls)/test_num, sum(ade4_ls)/test_num))
    print("Avg FDE:", sum(fde_ls) / test_num)
    print("Avg Coll:", sum(coll_ls) / test_num)
    # print("Avg AMD:", sum(amd_ls) / test_num)
    # print("Avg MDE:", sum(kde_ls) / test_num)
    # print("Avg AMV:", sum(eig_ls) / test_num)

    ### Visualize the best one in all trials
    if opt.is_visualize:
        visualize(raw_data_dic_, rn_pred_list, gt_list)

    # if opt.is_rn:
    #     rn_visualize(rn_pred_list[0], gt_list[0])
    #     # rn_visualize(rn_pred_list[1], gt_list[1])
    #     # rn_visualize(rn_pred_list[2], gt_list[2])
