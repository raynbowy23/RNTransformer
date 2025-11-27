import os
import argparse
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pickle
import torch
from torch_geometric.loader import DataLoader

from utils.metrics import * 
from data_loader import TrajectoryDataset, RoadNetworkDataset
from models.TrajectoryModel import *

parser = argparse.ArgumentParser(description="Parameter Settings for Training")

parser.add_argument('--pretrained_file', default="model_9.pt",
                help="Path to pretrained file which wants to be validated")

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
parser.add_argument('--is_rn_preprocessed', action="store_true", default=False,
                help="If RoadNetwork preprocessed file exists")
parser.add_argument('--load_preprocessed', action="store_true", default=False,
                help="If RoadNetwork preprocessed file exists")
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
parser.add_argument('--rn_uid', default=0, type=int,
                help="Unique ID for Road Network model")

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

model_path = Path(opt.pretrained_dir, opt.model_name, opt.pretrained_file) # '/val_best.pth'

# Data prep     
obs_seq_len = opt.num_timesteps_in
pred_seq_len = opt.num_timesteps_out

def load_data(opt):
    dataset_dir = Path(opt.dataset_dir, opt.dataset)

    print(obs_seq_len, pred_seq_len)

    if opt.is_rn:
        out_list = [1, 4, 8]
    else:
        out_list = [opt.num_timesteps_out]

    test_dataset = TrajectoryDataset(
            dataset_dir,
            sdd_loc=opt.sdd_loc,
            in_channels=opt.num_timesteps_in,
            out_channels=opt.num_timesteps_out,
            agg_frame=opt.agg_frame,
            skip=opt.skip,
            grid=opt.grid,
            norm_lap_matr=True,
            is_preprocessed=opt.is_preprocessed,
            dataset=opt.dataset,
            train_mode='test',
            is_rn=opt.is_rn,
            is_normalize=opt.is_normalize)

    test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0)

    test_rn_loader_list = []
    for i in range(len(out_list)):

        test_rn_dataset = RoadNetworkDataset(
                dataset_dir,
                sdd_loc=opt.sdd_loc,
                in_channels=opt.rn_num_timesteps_in,
                out_channels=opt.rn_num_timesteps_out,
                rn_out_channels=out_list[i],
                agg_frame=opt.agg_frame,
                skip=opt.skip,
                grid=opt.grid,
                is_preprocessed=opt.is_rn_preprocessed,
                dataset=opt.dataset,
                train_mode='val')

        if opt.is_rn:
            test_rn_loader_list.append(test_rn_dataset)

    if opt.is_rn:
        return test_loader, test_rn_loader_list
    else:
        return test_loader

h = None
out_list = [1, 4, 8]
num_nodes = 0

if opt.is_rn:
    test_loader, test_rn_dataset = load_data(opt)

    test_rn_loaders = [
        DataLoader(ds, batch_size=1, shuffle=False)
        for ds in test_rn_dataset
    ]
else:
    test_loader = load_data(opt)

### Model saving directory
print("===> Save model to %s" % opt.pretrained_dir)

os.makedirs(opt.pretrained_dir, exist_ok=True)

if opt.is_rn:
    print("===== Initializing model for time horizon =====")
    num_nodes = len(next(iter(test_rn_loaders[0])).x)
    print(num_nodes)

### Model setup
print("===== Initializing model for trajectory prediction =====")
if opt.is_rn:
    out_string = ''
    for out in out_list:
        out_string += str(out) + '_'
    rn_checkpoint_path = Path(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}epoch{}.pt'.format(opt.rn_uid, opt.grid, out_string, 10))

    rn_config = {
        'node_features': 8,
        'num_nodes': len(next(iter(test_rn_dataset[0])).x),
        'periods': opt.num_timesteps_in,
        'output_dim_list': [1, 4, 8],
        'device': device
    }
    
    model_rn = RNTransformer(**rn_config).to(device)
    model_rn.load_state_dict(torch.load(rn_checkpoint_path, weights_only=True))

    ### Freeze the parameters
    for param in model_rn.parameters():
        param.requires_grad = False
else:
    model_rn = None

model = TrajectoryModel(opt, num_nodes=num_nodes, out_list=out_list, device=device).to(device)

# Check if pretrained model exists
if opt.pretrained_epoch == None:
    if opt.is_rn:
        model.load_state_dict(torch.load(Path(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_rn_best.pt'.format(opt.uid)), map_location='cuda:0'), strict=False)
    else:
        model.load_state_dict(torch.load(Path(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_best.pt'.format(opt.uid))), strict=False)
else:
    if opt.dataset == 'sdd':
        d_name = 'sdd'
    else:
        d_name = 'eth'
    if opt.is_rn:
        model.load_state_dict(torch.load(Path(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_grid{}_epoch{}.pt'.format(opt.uid, opt.grid, opt.pretrained_epoch)), map_location='cuda:0'), strict=False)
    else:
        model.load_state_dict(torch.load(Path(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_{}.pt'.format(opt.uid, opt.pretrained_epoch))), strict=False)

total_param = 0
for param_tensor in model.state_dict():
    total_param += np.prod(model.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)

stats = Path(opt.pretrained_dir, opt.model_name, opt.dataset, 'constant_metrics.pkl')
with open(stats, 'rb') as f:
    cm = pickle.load(f)
print("Stats:", cm)

print(f'Pretrained Epoch: {opt.pretrained_epoch}')

@torch.no_grad()
def precompute_rn_features(model_rn, test_rn_loaders):
    """Pre-compute RN embeddings for all trajectory data"""
    feature_cache = {}
    logger.info("Pre-computing RN features for all trajectory data")
    
    if opt.load_preprocessed:
        with open(Path(opt.pretrained_dir, 'road_network', opt.dataset, 'feature_cache_vis.pkl'), 'rb') as f:
            feature_cache = pickle.load(f)
    else:
        # Create a lightweight dataloader just for RN computation
        for idx, batches in tqdm(enumerate(zip(*test_rn_loaders)), desc="Precomputing RN features"):
            x_list = [batch.x.to(device) for batch in batches]
            gt_list = [batch.y.to(device) for batch in batches]
            edge_attr_list = [batch.edge_attr.to(device) for batch in batches]
            edge_index_list = [batch.edge_index.to(device) for batch in batches]

            # x_list, edge_index_list, edge_attr_list = batch.x, batch.edge_index, batch.edge_attr
            rn_pred, rn_embed = model_rn(x_list, edge_index_list, edge_attr_list)
            # logger.info(rn_pred)
            feature_cache[idx] = (rn_pred, rn_embed.detach().cpu(), gt_list)

        with open(Path(opt.pretrained_dir, 'road_network', opt.dataset, 'feature_cache_vis.pkl'), 'wb') as f:
            pickle.dump(feature_cache, f)

    return feature_cache

def ade_per_node(pred, target):
    """
    pred:   (K, 1, T, 2)
    target: (1, T, 2)
    
    Returns:
        ade_per_sample: (K,)   # ADE for each of K predictions
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    # Remove singleton ped dim → becomes (K,T,2)
    pred = pred[:, 0]
    target = target[0]

    # Euclidean error per timestep, per prediction
    diff = np.sqrt(((pred - target)**2).sum(axis=-1))   # (K,T)

    # ADE per prediction
    ade = diff.mean(axis=-1)   # (K,)

    return ade

def visualize_feature_heatmap_3d(model, test_loader, rn_feature_cache, device, top_k=30):
    """
    PRODUCES:
        Heatmaps (No RN, With RN, Diff)
        Matching 3D surface plots:
            - Z = activation (or diff)
            - Color = same activation (or diff)
            - X = feature dim
            - Y = pedestrian index (sorted)
    """

    print(f"\n{'='*20} Generating Feature Heatmaps + 3D Surfaces {'='*20}")

    # Register hook
    features = {}
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach().cpu()
        return hook

    model_type = getattr(model, 'model_name', '')
    if 'social_stgcnn' in model_type:
        target_layer = model.model_loc.prelus[-1]
    elif 'social_implicit' in model_type:
        target_layer = model.model_loc.implicit_cells[0].ped.tpcnn
    else:
        target_layer = list(model.model_loc.modules())[-1]

    hook_handle = target_layer.register_forward_hook(get_activation('hidden'))

    # Collect features
    feats_rn, feats_no_rn = [], []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx > 0:
                break

            loc_data = [t.to(device) if not isinstance(t, list) else t for t in batch]
            obs_traj, _, _, _, _, _, V_obs, A_obs, _, _, _, _, ped_list = loc_data

            V_obs_tmp = V_obs.permute(0,3,1,2)

            # -- With RN
            rn_data = rn_feature_cache.get(idx, None)
            model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=rn_data)
            feats_rn.append(features['hidden'].clone().numpy())

            # -- Without RN
            model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=None)
            feats_no_rn.append(features['hidden'].clone().numpy())

    hook_handle.remove()

    # Format features
    f_rn = np.concatenate(feats_rn, axis=0)
    f_no = np.concatenate(feats_no_rn, axis=0)

    # Handle 4D activations → flatten
    if f_rn.ndim == 4:
        f_rn = f_rn.transpose(0,3,1,2)
        f_no = f_no.transpose(0,3,1,2)

    f_rn = f_rn.reshape(f_rn.shape[0], -1)
    f_no = f_no.reshape(f_no.shape[0], -1)

    diff = np.abs(f_rn - f_no)

    f_rn_sub = f_rn
    f_no_sub = f_no
    diff_sub = diff

    # For heatmap vmin/vmax
    vmin = min(f_rn_sub.min(), f_no_sub.min())
    vmax = max(f_rn_sub.max(), f_no_sub.max())

    # Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    sns.heatmap(f_no_sub, ax=axes[0], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Without RN")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Pedestrian Index (sorted)")

    sns.heatmap(f_rn_sub, ax=axes[1], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("With RN")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("")

    sns.heatmap(diff_sub, ax=axes[2], cmap="magma")
    axes[2].set_title("Difference |With–Without|")
    axes[2].set_xlabel("Feature Index")
    axes[2].set_ylabel("")

    plt.tight_layout()
    plt.savefig("figures/feature_heatmap_by_distance.png", dpi=300)
    plt.show()

    print("✓ Saved heatmaps")

    # 3D SURFACE PLOT GENERATOR
    def plot_3d_scatter(Z_values, title, save_name, cmap='viridis'):
        """
        3D scatter plot for feature activations.
        
        Z_values: array shape (N_nodes, N_feat)
                OR (top_k, num_features)

        Each point corresponds to:
        X = feature index
        Y = pedestrian index (node index)
        Z = activation
        color = activation
        """
        N_nodes, N_feat = Z_values.shape

        # Build X, Y coordinates for scatter
        X, Y = np.meshgrid(np.arange(N_feat), np.arange(N_nodes))

        # Flatten for scatter input
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z_values.flatten()

        # Normalize color
        Z_norm = (Z_flat - Z_flat.min()) / (Z_flat.max() - Z_flat.min() + 1e-8)

        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(projection='3d')

        scatter = ax.scatter(
            X_flat,
            Y_flat,
            Z_flat,
            c=Z_norm,
            cmap=cmap,
            s=18,        # point size
            alpha=0.9,
            linewidth=0
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Feature Index (X)")
        ax.set_ylabel("Pedestrian Index (Y)")
        ax.set_zlabel("Activation (Z)", fontsize=12)

        # Colorbar
        m = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Z_flat.min(), vmax=Z_flat.max()))
        m.set_array(Z_flat)
        fig.colorbar(m, ax=ax, shrink=0.5, label="Activation Value")

        # Camera angle
        ax.view_init(elev=35, azim=-45)

        plt.tight_layout()
        plt.savefig(f"figures/{save_name}.png", dpi=300)
        plt.show()

        print(f"✓ Saved 3D scatter plot: {save_name}.png")


    # Generate the 3D plots
    plot_3d_scatter(
        f_no_sub,
        "3D Feature Surface — WITHOUT RN\nZ = Activation, Color = Activation",
        "3d_feature_without_rn",
        cmap='magma'
    )

    plot_3d_scatter(
        f_rn_sub,
        "3D Feature Surface — WITH RN\nZ = Activation, Color = Activation",
        "3d_feature_with_rn",
        cmap='magma'
    )

    plot_3d_scatter(
        diff_sub,
        "3D Feature Surface — RN Sensitivity (|With–Without|)\nZ = Diff, Color = Diff",
        "3d_feature_diff_rn",
        cmap='magma'
    )


def visualize_feature_heatmap(model, test_loader, rn_feature_cache, device, top_k=30):
    """
    Generates a heatmap comparing hidden feature activations.
    
    Args:
        top_k (int): Number of pedestrians to visualize (to keep the plot readable).
    """
    print(f"\n{'='*20} Generating Feature Heatmaps {'='*20}")
    
    # Hook Registration
    features = {}
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach().cpu()
        return hook

    # Detect model and register hook
    hook_handle = None
    model_type = getattr(model, 'model_name', '')
    
    if 'social_stgcnn' in model_type:
        target_layer = model.model_loc.prelus[-1]
    elif 'social_implicit' in model_type:
        target_layer = model.model_loc.implicit_cells[0].ped.tpcnn
    else:
        target_layer = list(model.model_loc.modules())[-1]

    hook_handle = target_layer.register_forward_hook(get_activation('hidden'))
    
    feats_rn = []
    feats_no_rn = []
    
    # Data Collection (One pass is usually enough for heatmaps)
    model.eval()
    with torch.no_grad():
        # Just take the first batch to get a sample of pedestrians
        for idx, batch in enumerate(test_loader):
            if idx > 0: break 
            
            loc_data = [t.to(device) if not isinstance(t, list) else t for t in batch]
            obs_traj, _, _, _, _, _, V_obs, A_obs, _, _, _, _, ped_list = loc_data
            V_obs_tmp = V_obs.permute(0, 3, 1, 2)

            # With RN
            rn_data = rn_feature_cache[idx] if idx in rn_feature_cache else None
            model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=rn_data)
            feats_rn.append(features['hidden'].clone().numpy())

            # Without RN
            model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=None)
            feats_no_rn.append(features['hidden'].clone().numpy())

    hook_handle.remove()

    # Formatting
    # Concatenate and Flatten: [Peds, Features]
    f_rn = np.concatenate(feats_rn, axis=0)
    f_no_rn = np.concatenate(feats_no_rn, axis=0)

    # Handle 3D/4D shapes -> Flatten to 2D [N_Peds, N_Dim]
    if f_rn.ndim == 4: # STGCNN [B, C, T, N] -> [B, N, C, T]
        f_rn = f_rn.transpose(0, 3, 1, 2)
        f_no_rn = f_no_rn.transpose(0, 3, 1, 2)
    
    # Flatten all feature dims
    f_rn = f_rn.reshape(f_rn.shape[0], -1)
    f_no_rn = f_no_rn.reshape(f_no_rn.shape[0], -1)

    # Calculate Absolute Difference
    diff = np.abs(f_rn - f_no_rn)
    
    # Sorting for Visualization
    # Sort pedestrians by the mean magnitude of change (Difference)
    # This puts the pedestrians most affected by the Road Network at the top
    mean_diff = diff.mean(axis=1)
    sorted_indices = np.argsort(mean_diff)[::-1][:top_k] # Top K affected
    
    # Select subsets
    f_rn_sub = f_rn[sorted_indices]
    f_no_rn_sub = f_no_rn[sorted_indices]
    diff_sub = diff[sorted_indices]

    # Normalize for visualization (shared scale between RN/NoRN)
    # usually typically between -1 and 1 or 0 and 1 depending on activation
    vmin = min(f_rn_sub.min(), f_no_rn_sub.min())
    vmax = max(f_rn_sub.max(), f_no_rn_sub.max())

    #  Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    
    # Plot A: Without RN
    sns.heatmap(f_no_rn_sub, ax=axes[0], cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)
    axes[0].set_title("Features: Without RN", fontsize=14)
    axes[0].set_ylabel("Pedestrian Samples (Sorted by Impact)", fontsize=12)
    axes[0].set_xlabel("Feature Dimensions", fontsize=12)

    # Plot B: With RN
    sns.heatmap(f_rn_sub, ax=axes[1], cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)
    axes[1].set_title("Features: With RN", fontsize=14)
    axes[1].set_yticks([]) # Hide y-axis labels for middle plot
    axes[1].set_xlabel("Feature Dimensions", fontsize=12)

    # Plot C: Difference (Impact)
    # We use a different colormap (e.g., magma) to highlight activation differences
    sns.heatmap(diff_sub, ax=axes[2], cmap="magma", cbar_kws={'label': 'Activation Magnitude'})
    axes[2].set_title("Absolute Difference (|With - Without|)", fontsize=14)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Feature Dimensions", fontsize=12)

    plt.tight_layout()
    save_path = "figures/feature_heatmap_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Heatmap saved to {save_path}")
    plt.show()


def visualize_rn_feature_effects(model, test_loader, rn_feature_cache, device, top_k=30):
    """
    Visualizes RN effects on hidden activations using:
    (1) Scatter: RN Impact vs Travel Distance
    (2) 3D surface: Feature index x node index, Z=ADE, color = RN impact
    (3) Heatmap: Node x Feature (sorted by ADE)
    (4) Feature Sensitivity: mean RN impact per feature, which features change most
    
    Assumptions:
      - Non-LSTM model (social_stgcnn / social_implicit style)
      - model(...) with precomputed_rn returns (_, traj_pred, _)
      - traj_pred after permute has shape (K, P, T, 2)
      - Hidden hook output can be reshaped to (N_nodes, N_feat)
      - N_nodes == K * P
    """

    print(f"\n{'='*20} Generating RN Feature Visualizations {'='*20}")

    ## Register hook to capture hidden features
    features = {}
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach().cpu()
        return hook

    model_type = getattr(model, 'model_name', '')
    if 'social_stgcnn' in model_type:
        target_layer = model.model_loc.prelus[-1]
    elif 'social_implicit' in model_type:
        target_layer = model.model_loc.implicit_cells[0].ped.tpcnn
    else:
        target_layer = list(model.model_loc.modules())[-1]

    hook_handle = target_layer.register_forward_hook(get_activation('hidden'))

    ## Run one batch to collect features with/without RN
    feats_rn = []
    feats_no_rn = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx > 0:
                break

            loc_data = [t.to(device) if not isinstance(t, list) else t for t in batch]
            obs_traj, _, obs_traj_rel, _, _, _, V_obs, A_obs, V_tr, _, _, _, ped_list = loc_data

            V_obs_tmp = V_obs.permute(0, 3, 1, 2) # (B,C,T,N)

            # with RN
            rn_data = rn_feature_cache.get(idx, None)
            _, traj_pred, _ = model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=rn_data)
            feats_rn.append(features['hidden'].clone().numpy())

            # without RN
            _, traj_pred_no_rn, _ = model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=None)
            feats_no_rn.append(features['hidden'].clone().numpy())

    hook_handle.remove()

    f_rn = np.concatenate(feats_rn, axis=0)
    f_no_rn = np.concatenate(feats_no_rn, axis=0)

    if f_rn.ndim == 4:
        f_rn = f_rn.transpose(0, 3, 1, 2)
        f_no_rn = f_no_rn.transpose(0, 3, 1, 2)

    f_rn = f_rn.reshape(f_rn.shape[0], -1)
    f_no_rn = f_no_rn.reshape(f_no_rn.shape[0], -1)

    N_nodes, N_feat = f_rn.shape
    print(N_nodes, N_feat)

    # For non-social_lstm models: permute to (K,P,T,2)
    if opt.model_name != "social_lstm":
        traj_pred = traj_pred.permute(0, 2, 3, 1) # (K, P, T, 2)
        traj_pred_no_rn = traj_pred_no_rn.permute(0, 2, 3, 1)
    # print(traj_pred.shape, traj_pred_no_rn.shape, V_tr_batch.shape)

    ## Compute ADE per pedestrian
    num_of_peds = obs_traj_rel.shape[1]
    ade_list_rn = []
    ade_list_no = []
    for n in range(num_of_peds):
        pred = traj_pred[:, n:n+1] # (K,1,T,2)
        pred_no = traj_pred_no_rn[:, n:n+1]
        target = V_tr[:, n:n+1] # (1,1,T,2)

        ade_rn = ade_per_node(pred, target) # (K,)
        ade_no = ade_per_node(pred_no, target) # (K,)
        ade_list_rn.append(ade_rn)
        ade_list_no.append(ade_no)
    
    # Stack into shape (P*K,)
    loss_vector = np.concatenate(ade_list_rn)

    ade_rn = np.concatenate(ade_list_rn)   # (K*P,)
    ade_no = np.concatenate(ade_list_no)   # (K*P,)

    # Hidden features → reshape to (N_nodes, N_feat)
    if f_rn.ndim == 4:
        f_rn = f_rn.transpose(0, 3, 1, 2)
        f_no_rn = f_no_rn.transpose(0, 3, 1, 2)

    f_rn = f_rn.reshape(f_rn.shape[0], -1)
    f_no_rn = f_no_rn.reshape(f_no_rn.shape[0], -1)

    # RN impact
    diff = np.abs(f_rn - f_no_rn)         # (N_ped, N_feat)
    mean_diff = diff.mean(axis=1)         # scalar per pedestrian
    feature_importance = diff.mean(axis=0)  # scalar per feature

    # Plot 1: Scatter — RN Impact vs ADE
    plt.figure(figsize=(8, 6))
    plt.scatter(loss_vector, mean_diff, c=mean_diff, cmap='magma', s=40)
    plt.xlabel("ADE per node (ped×sample)")
    plt.ylabel("Mean RN Impact (|with - without|)")
    plt.title("RN Impact vs Prediction Error (ADE)")
    plt.colorbar(label="RN Impact Magnitude")
    plt.tight_layout()
    plt.savefig("figures/rn_scatter_ade_vs_impact.png", dpi=300)
    print("Saved scatter: rn_scatter_ade_vs_impact.png")
    plt.show()

    # 3D Plot: Node × Feature × ADE (color = RN impact)
    # Z = ADE replicated over all features
    Z_rn = np.repeat(ade_rn[:, None], N_feat, axis=1)
    Z_no = np.repeat(ade_no[:, None], N_feat, axis=1)

    C_rn = (Z_rn - Z_rn.min()) / (Z_rn.max() - Z_rn.min() + 1e-8)
    C_no = (Z_no - Z_no.min()) / (Z_no.max() - Z_no.min() + 1e-8)

    X, Y = np.meshgrid(np.arange(N_feat), np.arange(N_nodes))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    surf = ax.plot_surface(
        X, Y, Z_no,
        facecolors=plt.cm.magma(C_no),
        rstride=1, cstride=1,
        linewidth=0, antialiased=False,
    )

    ax.set_xlabel("Feature Dimension Index (X)")
    ax.set_ylabel("Node Index (ped × sample) (Y)")
    ax.set_zlabel("Prediction Error (ADE) (Z)")
    ax.set_title("RN Impact Landscape\nZ = ADE, Color = RN Feature Impact")

    m = plt.cm.ScalarMappable(cmap='magma')
    m.set_array(C_no)
    fig.colorbar(m, ax=ax, shrink=0.6, label='Activation Difference (|With - Without RN|)')

    ax.view_init(elev=35, azim=-45)

    plt.tight_layout()
    plt.savefig("figures/rn_3d_node_feat_ade.png", dpi=300)
    print("Saved 3D surface: rn_3d_node_feat_ade.png")
    plt.show()

    # Heatmap: Top-K nodes by ADE
    sorted_idx = np.argsort(loss_vector)[-top_k:]  # highest ADE
    diff_sorted = diff[sorted_idx]

    plt.figure(figsize=(12, 6))
    sns.heatmap(diff_sorted, cmap="magma", cbar=True)
    plt.xlabel("Feature Index")
    plt.ylabel("Nodes (Top-K by ADE)")
    plt.title("RN Feature Impact Heatmap (Top-K Hard Nodes)")
    plt.tight_layout()
    plt.savefig("figures/rn_heatmap_by_ade.png", dpi=300)
    print("Saved heatmap: rn_heatmap_by_ade.png")
    plt.show()

    # Feature sensitivity curve
    plt.figure(figsize=(10, 5))
    plt.plot(feature_importance, marker='o')
    plt.xlabel("Feature Index")
    plt.ylabel("Mean RN Impact Across Nodes")
    plt.title("Which Features Change the Most Due to RN?")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/rn_feature_sensitivity.png", dpi=300)
    print("Saved feature sensitivity: rn_feature_sensitivity.png")
    plt.show()

    print("\nAll RN visualizations completed.\n")

def visualize_rn_effect(model, test_loader, rn_feature_cache, device, batches_to_vis=5):
    """
    Visualizes the shift in hidden feature distribution with and without Road Network info.
    """
    print(f"\n{'='*20} Visualizing Hidden Features {'='*20}")
    
    # --- 1. Define and Register Hook ---
    features = {}
    def get_activation(name):
        def hook(model, input, output):
            # output shape is usually [Batch, Channels, Time, Nodes] for STGCNN
            features[name] = output.detach().cpu()
        return hook

    # Automatically find the layer to hook based on model type
    hook_handle = None
    if opt.model_name == 'social_stgcnn':
        # Hook the last PReLU layer of the temporal processing branch (before decoder)
        # SocialStgcnn structure: st_gcns -> tpcnns -> prelus -> tpcnn_output (decoder)
        target_layer = model.model_loc.prelus[-1]
        hook_handle = target_layer.register_forward_hook(get_activation('hidden'))
        print("Hook registered on: SocialStgcnn (prelus[-1])")
        
    elif opt.model_name == 'social_implicit':
        # Hook the local stream of the first social zone
        target_layer = model.model_loc.implicit_cells[0].ped.tpcnn
        hook_handle = target_layer.register_forward_hook(get_activation('hidden'))
        print("Hook registered on: SocialImplicit (implicit_cells[0])")

    if hook_handle is None:
        print("Could not register hook automatically. Please check model_name.")
        return

    model.eval()
    
    feats_with_rn = []
    feats_without_rn = []
    
    # --- 2. Run Inference ---
    print(f"Collecting features from {batches_to_vis} batches...")
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= batches_to_vis:
                break
            
            # Prepare Data
            loc_data = [t.to(device) if not isinstance(t, list) else t for t in batch]
            # Unpack typical batch (adjust if your loader returns different tuple)
            obs_traj, _, _, _, _, _, V_obs, A_obs, _, _, _, _, ped_list = loc_data
            
            # [Batch, 2, Time, Nodes]
            V_obs_tmp = V_obs.permute(0, 3, 1, 2)

            # Pass 1: With Road Network
            # Fetch precomputed RN features if available
            rn_data = rn_feature_cache[idx] if (rn_feature_cache and idx in rn_feature_cache) else None
            
            # Run model (RN branch active)
            model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=rn_data)
            
            if 'hidden' in features:
                # Store features (Clone to avoid overwrite)
                feats_with_rn.append(features['hidden'].clone().numpy())

            # Pass 2: Without Road Network
            # Force precomputed_rn=None to simulate 'no RN' (TrajectoryModel logic handles this)
            model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=None)
            
            if 'hidden' in features:
                feats_without_rn.append(features['hidden'].clone().numpy())

    # Clean up hook
    hook_handle.remove()

    # --- 3. Process and Visualize ---
    if not feats_with_rn:
        print("No features collected.")
        return

    # Concatenate all batches: Shape [Total_Batch, Channels, Time, Nodes]
    f_rn = np.concatenate(feats_with_rn, axis=0)
    f_no_rn = np.concatenate(feats_without_rn, axis=0)
    
    print(f"Raw Feature Shape: {f_rn.shape}")

    n_samples = f_rn.shape[0]
    f_rn_flat = f_rn.reshape(n_samples, -1)
    f_no_rn_flat = f_no_rn.reshape(n_samples, -1)
    print(f"Flattened to 3D logic: {f_rn_flat.shape}")

    combined = np.concatenate([f_rn_flat, f_no_rn_flat], axis=0)
    labels = np.array([0] * len(f_rn_flat) + [1] * len(f_no_rn_flat))
    
    # --- 4. Visualization ---
    # Combine for PCA
    combined = np.concatenate([f_rn_flat, f_no_rn_flat], axis=0)
    labels = np.array([0] * len(f_rn_flat) + [1] * len(f_no_rn_flat))
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot "With RN" (Blue)
    plt.scatter(reduced[labels==0, 0], reduced[labels==0, 1], 
                c='dodgerblue', alpha=0.6, label='With RN', edgecolors='k', linewidth=0.5, s=50)
    
    # Plot "Without RN" (Red)
    plt.scatter(reduced[labels==1, 0], reduced[labels==1, 1], 
                c='crimson', alpha=0.6, label='Without RN', edgecolors='k', linewidth=0.5, s=50)
    
    # Draw connecting lines to show shift
    # We limit lines to 100 samples to avoid clutter if the dataset is huge
    limit_lines = 100
    indices = np.arange(len(f_rn_flat))
    if len(indices) > limit_lines:
        np.random.shuffle(indices)
        indices = indices[:limit_lines]

    for i in indices:
        start = reduced[i]              # With RN
        end = reduced[i + len(f_rn_flat)] # Without RN (offset by length of first set)
        plt.plot([start[0], end[0]], [start[1], end[1]], color='gray', alpha=0.3, linestyle='-')    
    
    plt.legend(fontsize=12)
    plt.title(f'Hidden Feature Space: Impact of Road Network\nModel: {opt.model_name}', fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = 'figures/rn_feature_impact.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    rn_feature_cache = precompute_rn_features(model_rn, test_rn_loaders) if opt.is_rn else None
    visualize_rn_effect(model, test_loader, rn_feature_cache, device)
    visualize_rn_feature_effects(model, test_loader, rn_feature_cache, device)
    visualize_feature_heatmap(model, test_loader, rn_feature_cache, device, top_k=40)
    visualize_feature_heatmap_3d(model, test_loader, rn_feature_cache, device, top_k=40)