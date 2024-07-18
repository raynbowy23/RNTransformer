import math
from sklearn.mixture import GaussianMixture
from math import sqrt, exp
from scipy.special import erf
from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
import numpy as np

## Average Displacement Error
def ade_loss(predAll, targetAll, count_):
    All = count_
    sum_all = 0
    for s in range(1, count_+1):
        pred = predAll[0][:,:s,:]
        target = targetAll[0][:,:s,:]
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i,t,0] - target[i,t,0])**2 + (pred[i,t,1] - target[i,t,1])**2)
                
        sum_all += sum_ / (N * T)
        
    return sum_all / All 

#TODO: Implementation: Average Absolute Heading Error

# Average Displacement Error
def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    sum_all1 = 0 
    sum_all2 = 0 
    sum_all3 = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:], 0, 1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:], 0, 1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = [0 for _ in range(4)]
        # sum_ = 0
        for i in range(N):
            cnt = 0
            for t in range(T):
                sum_[0] += math.sqrt((pred[i,t,0] - target[i,t,0])**2 + (pred[i,t,1] - target[i,t,1])**2)
                if 0 <= t < int(T/3):
                    sum_[1] += math.sqrt((pred[i,t,0] - target[i,t,0])**2 + (pred[i,t,1] - target[i,t,1])**2)
                elif int(T/3) <= t < int(T/3*2):
                    sum_[2] += math.sqrt((pred[i,t,0] - target[i,t,0])**2 + (pred[i,t,1] - target[i,t,1])**2)
                else:
                    sum_[3] += math.sqrt((pred[i,t,0] - target[i,t,0])**2 + (pred[i,t,1] - target[i,t,1])**2)
                
        sum_all += sum_[0] / (N * T)
        sum_all1 += sum_[1] / (N * T)
        sum_all2 += sum_[2] / (N * T)
        sum_all3 += sum_[3] / (N * T)
        
    # return sum_all / All
    return sum_all / All, sum_[1] / 12, sum_[2] / 12, sum_[3] / 12
'''


def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All
'''


# Final Displacement Error
def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:], 0, 1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1, T):
                sum_ += math.sqrt((pred[i,t,0] - target[i,t,0])**2 + (pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_ / (N)

    return sum_all / All

def min_max_normalize(data):
    min_values, _ = torch.min(data, dim=0)
    max_values, _ = torch.max(data, dim=0)
    eps = 1e-10
    
    normalized_data = (data - min_values) / (max_values - min_values + eps)
    # if torch.equal(min_values, torch.tensor([0., 0.])) or torch.equal(max_values, torch.tensor([0., 0.])):
    #     normalized_data = (data - min_values) / (max_values - min_values + eps)
    # else:
    #     normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data, min_values, max_values

def min_max_unnormalize(normalized_data, min_values, max_values):
    unnormalized_data = normalized_data * (max_values - min_values) + min_values
    return unnormalized_data

def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)): 
            V[s, h, :] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s+1, ped, :], axis=0) + init_node[ped, :]

    return nodes_.squeeze()

def closer_to_zero(current, new_v):
    dec = min([(abs(current), current),(abs(new_v), new_v)])[1]
    if dec != current:
        return True
    else: 
        return False

def kl_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    assert len(V_pred.shape) == len(V_trgt.shape)
    if len(V_pred.shape) == 4:
        normx = V_trgt[:, :, :, 0] - V_pred[:, :, :, 0]
        normy = V_trgt[:, :, :, 1] - V_pred[:, :, :, 1]

        sx = torch.exp(V_pred[:, :, :, 2]) #sx
        sy = torch.exp(V_pred[:, :, :, 3]) #sy
        corr = torch.tanh(V_pred[:, :, :, 4]) #corr
    else:
        normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
        normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

        sx = torch.exp(V_pred[:, :, 2]) #sx
        sy = torch.exp(V_pred[:, :, 3]) #sy
        corr = torch.tanh(V_pred[:, :, 4]) #corr

    # Numerical stability
    epsilon = 1e-20
    
    sx = sx + epsilon
    sy = sy + epsilon
    sxsy = sx * sy

def multivariate_loss(V_pred, V_trgt, training=True):
    r"""Batch multivariate loss"""

    mu = V_trgt[:, :, :, 0:2] - V_pred[:, :, :, 0:2]
    mu = mu.unsqueeze(dim=-1)

    sx = V_pred[:, :, :, 2].exp()
    sy = V_pred[:, :, :, 3].exp()
    corr = V_pred[:, :, :, 4].tanh()

    cov = torch.zeros(V_pred.size(0), V_pred.size(1), V_pred.size(2), 2, 2).cuda()

    cov[:, :, :, 0, 0] = sx * sx
    cov[:, :, :, 0, 1] = corr * sx * sy
    cov[:, :, :, 1, 0] = corr * sx * sy
    cov[:, :, :, 1, 1] = sy * sy
    #cov = cov.clamp(min=-1e5, max=1e5)

    pdf = torch.exp(-0.5 * mu.transpose(-2, -1) @ cov.inverse() @ mu)
    pdf = pdf.squeeze() / torch.sqrt(((2 * np.pi) ** 2) * cov.det())

    if training:
        pdf[torch.isinf(pdf) | torch.isnan(pdf)] = 0

    epsilon = 1e-20
    loss = -pdf.clamp(min=epsilon).log()

    return loss.mean()

        
def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    assert len(V_pred.shape) == len(V_trgt.shape)
    if len(V_pred.shape) == 4:
        normx = V_trgt[:, :, :, 0] - V_pred[:, :, :, 0]
        normy = V_trgt[:, :, :, 1] - V_pred[:, :, :, 1]

        sx = torch.exp(V_pred[:, :, :, 2]) # sx
        sy = torch.exp(V_pred[:, :, :, 3]) # sy
        corr = torch.tanh(V_pred[:, :, :, 4]) # corr -> Rho
    else:
        normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
        normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

        sx = torch.exp(V_pred[:, :, 2]) # sx
        sy = torch.exp(V_pred[:, :, 3]) # sy
        corr = torch.tanh(V_pred[:, :, 4]) # corr

    # Numerical stability
    epsilon = 1e-20
    
    sxsy = sx * sy
    mux = torch.mean(normx)
    muy = torch.mean(normy)


    # z = ((normx - mux) / sx)**2 + ((normy - muy) / sy)**2 - 2 * corr * ((normx - mux) * (normy - muy) / sxsy)
    z = ((normx) / sx)**2 + ((normy) / sy)**2 - 2 * corr * ((normx) * (normy) / sxsy)
    negRho = 1 - corr**2 + epsilon

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))


    # Final PDF calculation
    result = result / denom

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result


_l1_mean = nn.L1Loss()


def cdist_cosine_sim(a, b, eps=1e-08):
    a_norm = a / torch.clamp(a.norm(dim=1)[:, None], min=eps)
    b_norm = b / torch.clamp(b.norm(dim=1)[:, None], min=eps)
    return torch.acos(
        torch.clamp(torch.mm(a_norm, b_norm.transpose(0, 1)),
                    min=-1.0 + eps,
                    max=1.0 - eps))


def implicit_likelihood_estimation_fast_with_trip_geo(V_pred, V_target):

    V_pred = V_pred.contiguous()

    diff = torch.abs(V_pred - V_target)

    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    _, indices = torch.sort(diff_sum)
    min_indx = indices[0]
    V_pred_min = V_pred[min_indx]
    V_target = V_target.squeeze()

    error = _l1_mean(V_pred_min, V_target)
    trip_loss = _l1_mean(V_pred_min, V_pred[indices[1]]) - _l1_mean(
        V_pred_min, V_pred[indices[-1]])
    # trip_loss = _l1_mean(V_pred_min, V_pred[0]) - _l1_mean(
    #     V_pred_min, V_pred[0])

    V_pred_min_ = V_pred_min.reshape(-1, 2)
    V_target_ = V_target.reshape(-1, 2)

    #Geometric distance length
    norm_loss = torch.abs(
        torch.cdist(V_pred_min_.unsqueeze(0), V_pred_min_.unsqueeze(0), p=2.0)
        - torch.cdist(V_target_.unsqueeze(0), V_target_.unsqueeze(0), p=2.0)
    ).mean()

    #Gemometric distance angle
    cos_loss = torch.abs(
        cdist_cosine_sim(V_pred_min_, V_pred_min_) -
        cdist_cosine_sim(V_target_, V_target_)).mean()

    # loss_store["l2"] += error.item()
    # loss_store["gl2"] += norm_loss.item()
    # loss_store["gcos"] += cos_loss.item()
    # loss_store["trip"] += trip_loss.item()

    # return error * norm_loss
    # return trip_loss
    return error + 0.00001 * norm_loss + 0.0001 * trip_loss #+ 0.0001 * cos_loss

#AMD / AMV
def calc_amd_amv(gt, pred):
    total = 0
    m_collect = []
    gmm_cov_all = 0
    for i in range(pred.shape[0]):  #per time step
        for j in range(pred.shape[1]):
            #do the method of finding the best bic
            temp = pred[i, j, :, :]

            gmm = get_best_gmm2(pred[i, j, :, :])
            center = np.sum(np.multiply(gmm.means_, gmm.weights_[:,
                                                                 np.newaxis]),
                            axis=0)
            gmm_cov = 0
            for cnt in range(len(gmm.means_)):
                gmm_cov += gmm.weights_[cnt] * (
                    gmm.means_[cnt] - center)[..., None] @ np.transpose(
                        (gmm.means_[cnt] - center)[..., None])
            gmm_cov = np.sum(gmm.weights_[..., None, None] * gmm.covariances_,
                             axis=0) + gmm_cov

            dist, _ = mahalanobis_d(
                center, gt[i, j], len(gmm.weights_), gmm.covariances_,
                gmm.means_, gmm.weights_
            )  #assume it will be the true value, add parameters

            total += dist
            gmm_cov_all += gmm_cov
            m_collect.append(dist)

    gmm_cov_all = gmm_cov_all / (pred.shape[0] * pred.shape[1])
    return total / (pred.shape[0] *
                    pred.shape[1]), None, None, m_collect, np.abs(
                        np.linalg.eigvals(gmm_cov_all)).max()


def mahalanobis_d(x, y, n_clusters, ccov, cmeans, cluster_p):  #ccov
    v = np.array(x - y)
    Gnum = 0
    Gden = 0
    for i in range(0, n_clusters):
        ck = np.linalg.pinv(ccov[i])
        u = np.array(cmeans[i] - y)
        val = ck * cluster_p[i]
        b2 = 1 / (v.T @ ck @ v)
        a = b2 * v.T @ ck @ u
        Z = u.T @ ck @ u - b2 * (v.T @ ck @ u)**2
        pxk = sqrt(np.pi * b2 / 2) * exp(-Z / 2) * (erf(
            (1 - a) / sqrt(2 * b2)) - erf(-a / sqrt(2 * b2)))
        Gnum += val * pxk
        Gden += cluster_p[i] * pxk
    G = Gnum / Gden
    mdist = sqrt(v.T @ G @ v)
    if np.isnan(mdist):
        # print(Gnum, Gden)
        '''
        print("is nan")
        print(v)
        print("Number of clusters", n_clusters)
        print("covariances", ccov)
        '''
        return 0, 0

    # print( "Mahalanobis distance between " + str(x) + " and "+str(y) + " is "+ str(mdist) )
    return mdist, G


def get_best_gmm(X):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(
        1, 7)  ## stop based on fit/small BIC change/ earlystopping
    cv_types = ['full']
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    return best_gmm


def get_best_gmm2(X):  #early stopping gmm
    lowest_bic = np.infty
    bic = []
    cv_types = ['full']  #changed to only looking for full covariance
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        p = 1  #Decide a value
        n_comps = 1
        j = 0
        while j < p and n_comps < 5:  # if hasn't improved in p times, then stop. Do it for each cv type and take the minimum of all of them
            gmm = GaussianMixture(n_components=n_comps,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                j = 0  #reset counter
            else:  #increment counter
                j += 1
            n_comps += 1

    bic = np.array(bic)
    return best_gmm


def kde_lossf(gt, pred):
    #(12, objects, samples, 2)
    # 12, 1600,1000,2
    kde_ll = 0
    kde_ll_f = 0
    n_u_c = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            temp = pred[i, j, :, :]
            n_unique = len(np.unique(temp, axis=0))
            if n_unique > 2:
                kde = gaussian_kde(pred[i, j, :, :].T)
                t = np.clip(kde.logpdf(gt[i, j, :].T), a_min=-20,
                            a_max=None)[0]
                kde_ll += t
                if i == (pred.shape[0] - 1):
                    kde_ll_f += t
            else:
                n_u_c += 1
    if n_u_c == pred.shape[0] * pred.shape[1]:
        return 0
    return -kde_ll / (pred.shape[0] * pred.shape[1])

def interpolate_traj(traj, num_interp=4):
    '''
    Add linearly interpolated points of a trajectory
    '''
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp+1):
        ratio = (i + 1) / (num_interp + 1)
        dense[:, i+1::num_interp+1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

    return dense

def compute_col(predicted_traj, predicted_trajs_all, thres=0.2):
    '''
    Input:
        predicted_trajs: predicted trajectory of the primary agents, [12, 2]
        predicted_trajs_all: predicted trajectory of all agents in the scene, [num_person, 12, 2]
    '''
    ph = predicted_traj.shape[0]
    num_interp = 4
    assert predicted_trajs_all.shape[0] > 1

    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)  # [num_person, 12 * num_interp]
    mask = distances[:, 0] > 0  # exclude primary agent itself
    return (distances[mask].min(axis=0) < thres)