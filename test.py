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
from models.simple import Simple
from data_loader import inDDatasetGraph, TrajectoryDataset
from models.TrajectoryModel import *
from models.LocalPedsTrajNet import *
import copy
from tqdm import tqdm
import argparse

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


ade_ls = [] 
fde_ls = [] 
kde_ls = []
amd_ls = []
eig_ls = []

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
            # out_list = [1]
            # out_list = [4, 8, 16]
            # out_list = [12]
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


def visualize(raw_data_dic_):
    '''
    Visualize trajectories (Prediction, Target, Observation)
    '''

    seq = [1, 6, 9, 11]
    # k = [0, 2, 12, 14]
    k = [0, 0, 0, 0]
    ped_id = 1

    _, ax = plt.subplots(2, 2) 

    print(len(raw_data_dic_))

    ax[0, 0].plot(raw_data_dic_[seq[0]]['obs'][:, ped_id, 0], raw_data_dic_[seq[0]]['obs'][:, ped_id, 1], color='g', lw=2., linestyle='-')
    ax[0, 0].plot(raw_data_dic_[seq[0]]['pred'][k[0]][:, ped_id, 0], raw_data_dic_[seq[0]]['pred'][k[0]][:, ped_id, 1], color='b', lw=2., linestyle='-')
    ax[0, 0].plot(raw_data_dic_[seq[0]]['trgt'][:, ped_id, 0], raw_data_dic_[seq[0]]['trgt'][:, ped_id, 1], color='r', lw=2., linestyle='-')
    sns.kdeplot(x=raw_data_dic_[seq[0]]['pred'][k[0]][:, ped_id, 0], y=raw_data_dic_[seq[0]]['pred'][k[0]][:, ped_id, 1], fill=True, ax=ax[0, 0])

    ax[0, 1].plot(raw_data_dic_[seq[1]]['obs'][:, ped_id, 0], raw_data_dic_[seq[1]]['obs'][:, ped_id, 1], color='g', lw=2., linestyle='-')
    ax[0, 1].plot(raw_data_dic_[seq[1]]['pred'][k[1]][:, ped_id, 0], raw_data_dic_[seq[1]]['pred'][k[1]][:, ped_id, 1], color='b', lw=2., linestyle='-')
    ax[0, 1].plot(raw_data_dic_[seq[1]]['trgt'][:, ped_id, 0], raw_data_dic_[seq[1]]['trgt'][:, ped_id, 1], color='r', lw=2., linestyle='-')
    sns.kdeplot(x=raw_data_dic_[seq[1]]['pred'][k[1]][:, ped_id, 0], y=raw_data_dic_[seq[1]]['pred'][k[1]][:, ped_id, 1], fill=True, ax=ax[0, 1])

    print(raw_data_dic_[seq[2]]['obs'][:, ped_id, 1])
    print(raw_data_dic_[seq[2]]['trgt'][:, ped_id, 1])
    print(raw_data_dic_[seq[2]]['pred'][0][:, ped_id, 1])
    ax[1, 0].plot(raw_data_dic_[seq[2]]['obs'][:, ped_id, 0], raw_data_dic_[seq[2]]['obs'][:, ped_id, 1], color='g', lw=2., linestyle='-')
    ax[1, 0].plot(raw_data_dic_[seq[2]]['pred'][k[2]][:, ped_id, 0], raw_data_dic_[seq[2]]['pred'][k[2]][:, ped_id, 1], color='b', lw=2., linestyle='-')
    ax[1, 0].plot(raw_data_dic_[seq[2]]['trgt'][:, ped_id, 0], raw_data_dic_[seq[2]]['trgt'][:, ped_id, 1], color='r', lw=2., linestyle='-')
    sns.kdeplot(x=raw_data_dic_[seq[2]]['pred'][k[2]][:, ped_id, 0], y=raw_data_dic_[seq[2]]['pred'][k[2]][:, ped_id, 1], fill=True, ax=ax[1, 0])

    ax[1, 1].plot(raw_data_dic_[seq[3]]['obs'][:, ped_id, 0], raw_data_dic_[seq[3]]['obs'][:, ped_id, 1], color='g', lw=2., linestyle='-')
    ax[1, 1].plot(raw_data_dic_[seq[3]]['pred'][k[3]][:, ped_id, 0], raw_data_dic_[seq[3]]['pred'][k[3]][:, ped_id, 1], color='b', lw=2., linestyle='-')
    ax[1, 1].plot(raw_data_dic_[seq[3]]['trgt'][:, ped_id, 0], raw_data_dic_[seq[3]]['trgt'][:, ped_id, 1], color='r', lw=2., linestyle='-')
    sns.kdeplot(x=raw_data_dic_[seq[3]]['pred'][k[3]][:, ped_id, 0], y=raw_data_dic_[seq[3]]['pred'][k[3]][:, ped_id, 1], fill=True, ax=ax[1, 1])
    # X, Y = torch.meshgrid(raw_data_dic_[0]['pred'][k][:, ped_id, 0], raw_data_dic_[0]['pred'][k][:, ped_id, 1])
    # plt.contourf(X, Y, pdf, cmap='virdis')


    plt.show()


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)

@torch.no_grad()
def test(test_loader, test_rn_dataset=None, KSTEPS=20):
    model.eval()
    # model_rn.eval()
    # ade_bigls = []
    ade_bigls = [[] for _ in range(4)]
    fde_bigls = []
    raw_data_dict = {}
    traj_pred = None
    loss = 0
    cnt = 0

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
                    loc_data.append(tensor.to(device))
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
                    loc_data.append(tensor.to(device))

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, min_rel_pred, max_rel_pred = loc_data

        num_of_objs = obs_traj_rel.shape[1]
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        if opt.is_rn:

            rn_edge_index = [batch[i].edge_index.to(device) for i in range(1, len(batch))]
            rn_edge_attr = [batch[i].edge_attr.to(device) for i in range(1, len(batch))]
            # rn_edge_attr = [batch[i].edge_attr for i in range(1, len(batch))]

            ### Model in one piece
            # rn_pred, traj_pred = model(V_obs_tmp, A_obs, x, rn_edge_index, rn_edge_attr, h_=traj_pred)
            rn_pred, out_loc, traj_pred = model(V_obs_tmp, A_obs, x, rn_edge_index, rn_edge_attr)
            out_loc = out_loc.permute(0, 2, 3, 1)
            # rn_pred, traj_pred, _ = model(V_obs_tmp, A_obs, x, rn_edge_index, rn_edge_attr)
        else:
            if (V_obs_tmp.size(-1) == 0):
                break
            rn_pred, traj_pred, _ = model(V_obs_tmp, A_obs)
            # rn_pred, traj_pred, _ = model(V_obs_tmp, A_obs)
            # traj_pred, _ = model(V_obs_tmp, A_obs.squeeze())
            traj_pred = traj_pred.permute(0, 2, 3, 1)

        traj_loss = graph_loss(traj_pred, V_tr)
        loss += traj_loss

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        traj_pred = traj_pred.squeeze() #.float()
        # traj_pred = out_loc.squeeze() #.float()
        num_of_objs = obs_traj_rel.shape[1]
        V_obs = V_obs[:, :num_of_objs, :]


        # For now I have my bi-variate parameters 
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]

        if opt.model_name != "social_implicit":
            traj_pred, V_tr = traj_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
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
        else:
            traj_pred, V_tr = traj_pred[:, :, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # traj_pred, V_tr = traj_pred[:, :, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        # sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())

        if opt.is_visualize and opt.dataset == "eth":
            ### Unnormalized
            V_x_rel_to_abs = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        else:
            ### Normalized
            # V_x_rel_to_abs = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
            V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                    V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        if opt.is_visualize and opt.dataset == "eth":
            ### Unnormalized
            V_y_rel_to_abs = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        else:
            ### Normalized
            # V_y_rel_to_abs = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
            V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                        V_x[-1, :, :].copy())

        
        raw_data_dict[i] = {}
        raw_data_dict[i]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[i]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[i]['pred'] = []

        ade_ls = [[[] for _ in range(num_of_objs)] for _ in range(4)]
        # ade_ls = [[] for _ in range(num_of_objs)]
        fde_ls = [[] for _ in range(num_of_objs)]


        b_samples = []
        for k in range(KSTEPS):

            if opt.model_name == "social_implicit":
                V_pred = traj_pred[k:k+1, ...].squeeze()
            else:
                V_pred = mvnormal.sample()

            if opt.is_visualize and opt.dataset == "eth":
                V_pred = min_max_unnormalize(V_pred, min_rel_pred, max_rel_pred)

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
    print(f'Loss: {traj_loss / cnt}')

    # ade1_ = sum(ade_bigls) / len(ade_bigls)
    ade1_ = sum(ade_bigls[0]) / len(ade_bigls[0])
    ade2_ = sum(ade_bigls[1]) / len(ade_bigls[1])
    ade3_ = sum(ade_bigls[2]) / len(ade_bigls[2])
    ade4_ = sum(ade_bigls[3]) / len(ade_bigls[3])
    fde_ = sum(fde_bigls) / len(fde_bigls)

    # return ade1_, fde_, raw_data_dict
    return ade1_, ade2_, ade3_, ade4_, fde_, raw_data_dict #, #sum(kde_loss) / len(kde_loss), sum(mabs_loss) / len(mabs_loss), sum(eig_collect) / len(eig_collect), raw_data_dict



if __name__ == '__main__':
    h = None
    out_list = [1, 4, 8]
    # out_list = [4, 8, 16]
    # out_list = [12]
    # out_list = [1]
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
    if opt.model_name == "Simple":
        model = Simple(in_channels=opt.num_timesteps_in, out_channels=opt.num_timesteps_out,
                    num_timesteps_in=opt.num_timesteps_in, is_horizontal_pred=opt.is_horizontal_pred).to(device)
    elif opt.model_name == "trajectory_model" or opt.model_name == "social_stgcnn" or opt.model_name == "social_implicit":
        model_rn = None
        model_loc = None
        # model_rn = RNTransformer(node_features=7, num_nodes=num_nodes, periods=opt.num_timesteps_in, output_dim_list=out_list, device=device).to(device)
        # model_path = osp.join(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_out_list{}_{}_{}_epoch{}.pt'.format(32, opt.grid, out_list[0], out_list[1], out_list[2], 50))
        # model_rn.load_state_dict(torch.load(model_path))
        # model_loc = social_stgcnn(1, 5, output_feat=5, seq_len=opt.num_timesteps_in, pred_seq_len=opt.num_timesteps_out, num_nodes=num_nodes).to(device)
        # model_loc.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_{}.pt'.format(200, 100))))
        model = TrajectoryModel(in_channels=opt.num_timesteps_in, out_channels=opt.num_timesteps_out,
                                 num_nodes=num_nodes, out_list=out_list, periods=opt.num_timesteps_in, 
                                depth=1, mlp_dim=128, device=device, is_rn=opt.is_rn, model_name=opt.model_name, model_loc=model_loc, model_rn=model_rn).to(device)
        # model = social_stgcnn(1, 5, output_feat=5, seq_len=opt.num_timesteps_in, pred_seq_len=opt.num_timesteps_out, num_nodes=num_nodes).to(device)

    # Check if pretrained model exists
    if opt.pretrained_epoch == None:
        if opt.is_rn:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_rn_best.pt'.format(opt.uid))))
        else:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_best.pt'.format(opt.uid))))
    else:
        if opt.is_rn:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, 'eth', '{}_model_grid{}_epoch{}.pt'.format(opt.uid, opt.grid, opt.pretrained_epoch))))
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
    print("Testing ....")
    test_num = 5
    for _ in range(test_num):
        if opt.is_rn:
            # ad, fd, kd, md, eg, raw_data_dic_= test(test_loader, test_rn_dataset)
            # ad, fd, raw_data_dic_= test(test_loader, test_rn_dataset)
            ad, ad2, ad3, ad4, fd, raw_data_dic_= test(test_loader, test_rn_dataset)
        else:
            # ad, fd, kd, md, eg, raw_data_dic_= test(test_loader)
            # ad, fd, raw_data_dic_= test(test_loader)
            ad, ad2, ad3, ad4, fd, raw_data_dic_= test(test_loader)
        ade_ = min(ade_, ad)
        fde_ = min(fde_, fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        # kde_ls.append(kd)
        # amd_ls.append(md)
        # eig_ls.append(eg)
        print("ADE:", ade_, " FDE:", fde_) #, "AMD:", md, "KDE:", kd, "AMV:", eg)

    print("*"*50)

    print("Avg ADE:", sum(ade_ls) / test_num)
    print("ADE First Period: {}, Second Period: {}, Third Period: {}".format(ad2, ad3, ad4))
    print("Avg FDE:", sum(fde_ls) / test_num)
    # print("Avg AMD:", sum(amd_ls) / test_num)
    # print("Avg MDE:", sum(kde_ls) / test_num)
    # print("Avg AMV:", sum(eig_ls) / test_num)

    ### Visualize the best one in all trials
    if opt.is_visualize:
        visualize(raw_data_dic_)
