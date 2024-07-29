import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
# plt.style.use('ssass.mplstyle')

def ids_to_coo(ids):
    '''
    Convert cluster assignments to a coo matrix

    Args:
    ids (np.array): the cluster assignments

    Returns:
    coo (np.array): the coo matrix
    '''
    
    coo = np.zeros((len(ids), len(ids)))
    for u in np.unique(ids):
        mask = ids == u
        coo[np.ix_(mask, mask)] = 1
    return coo

def viz_heatmap(title, iter, assigns_df = None, params_df = None, max_clusters=20, figures=True):
    '''
    Visualize the cluster assignments as a heatmap

    Args:
    title (str): the title number
    iter (int): the number of iterations
    assigns (pd.DataFrame): the cluster assignments
    params (pd.DataFrame): the cluster parameters
    max_clusters (int): the maximum number of clusters

    
    Returns:
    best_assigns (pd.DataFrame): the best cluster assignments
    best_params (pd.DataFrame): the best cluster parameters
    
    Saves best cluster assignments to {title}/{title}_best_assigns.csv
    Saves best cluster parameters to {title}/{title}_best_params.csv
    Saves a scatter plot of cluster parameters to {title}/{title}_params.png
    Saves a heatmap of the cluster assignments to {title}/{title}_assigns.png
    '''

    if iter < 510:
        if assigns_df is None or params_df is None:
            assigns = pd.read_csv(f"{title}/{title}_assigns.csv", header=None)
            params = pd.read_csv(f"{title}/{title}_params.tsv", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters))
        else:
            assigns = pd.read_csv(f"{assigns_df}", header=None)
            params = pd.read_csv(f"{params_df}", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters))

    else:
        if assigns_df is None or params_df is None:
            assigns = pd.read_csv(f"{title}/{title}_assigns.csv", header=None).iloc[500:].reset_index(drop=True)
            params = pd.read_csv(f"{title}/{title}_params.tsv", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters)).iloc[500:].reset_index(drop=True)
        else:
            assigns = pd.read_csv(f"{assigns_df}", header=None).iloc[500:].reset_index(drop=True)
            params = pd.read_csv(f"{params_df}", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters)).iloc[500:].reset_index(drop=True)
        

    params.fillna('0,0', inplace=True)
    split_params = params.applymap(lambda x: np.array(x.split(','), dtype=float))

    average_params = []

    for count, datapoint in enumerate(assigns):
        datapoint_params = []
        for index, assignments in enumerate(assigns[datapoint]):
            datapoint_params.append(split_params[assignments].iloc[index])
    
        # convert to numpy array for efficient array indexing
        datapoint_params = np.array(datapoint_params)

        # find average jump and average phasicity for current data point
        average_jump = np.mean(datapoint_params[:, 0])
        average_phasicity = np.mean(datapoint_params[:, 1])

        # final average parameter for data point
        final_datapoint_param_average = [average_jump, average_phasicity]

        # append to list of average parameters for all data points
        average_params.append(final_datapoint_param_average)

    average_params = np.array(average_params)
    # print(f'final params: {average_params}')

    coos = np.stack([ids_to_coo(x) for _, x in assigns.iterrows()])
    mean_coo = coos.mean(axis=0)
    dists = np.linalg.norm(coos - mean_coo, axis=(1, 2))
    min_dist_idx = np.argmin(dists)
    match_idxs = np.where(np.all(coos == coos[min_dist_idx], axis=(1, 2)))[0]

    # get the best assigns
    best_assigns =  pd.DataFrame(assigns.loc[min_dist_idx])
    best_assigns.columns = ['cluster id']

    # save params to file
    best_params = pd.DataFrame(average_params)
    best_params.columns = ['jump', 'phasicity']
    best_params.to_csv(f'{title}/{title}_best_params.csv')
    print(f'Found best parameters. Saving to file to {title}/{title}_best_params.csv')

    # rotate the df
    best_assigns = best_assigns.T

    # save best_assigns to csv file
    best_assigns.to_csv(f'{title}/{title}_best_assigns.csv')
    print(f'Found best assigns. Saving to file to {title}/{title}_best_assigns.csv')

    reorder = assigns.loc[min_dist_idx].sort_values().index

    if figures:
        # plot cluster assignments
        plt.imshow(mean_coo[np.ix_(reorder, reorder)], cmap="coolwarm")
        plt.colorbar()

        N = len(assigns.columns)

        plt.yticks(np.arange(N), reorder, fontsize=5)
        plt.xticks(np.arange(N), reorder, fontsize=5, rotation=90)
        plt.tight_layout()

        # Add chosen clustering in outline
        counts = np.bincount(best_assigns.iloc[0, :])
        start = -0.5
        for c in counts:
            rect = patches.Rectangle((start, start), c, c, linewidth=1, edgecolor='lime', facecolor='none')
            start += c
            plt.gca().add_patch(rect)

        if not os.path.exists(f'{title}'):
            os.makedirs(f'{title}')
        if not os.path.exists(f'{title}/{title}_assigns.png'):
            plt.savefig(f'{title}/{title}_assigns.png', dpi = 500)
        else:
            oldfile = f'{title}_assigns.png'
            os.remove(f'{title}/{oldfile}')
            plt.savefig(f'{title}/{title}_assigns.png', dpi = 500)
            
        plt.close()
        print(f'Saving cluster assignments heatmap to {title}/{title}_assigns.png')

        # make params figure
        if max(best_assigns.iloc[0, :]) > 19:
            color_palette = 'viridis'
        else:
            color_palette = 'tab20'

        min_alpha = 0.7
        max_alpha = 0.9

        unique_clusters = np.unique(best_assigns.iloc[0, :])
        num_clusters = len(unique_clusters)

        if num_clusters == 1:
            alphas = 0.9

        else:
            alphas = min_alpha + (best_assigns.iloc[0, :] - unique_clusters.min()) * (max_alpha - min_alpha) / (num_clusters - 1)

        plt.scatter(average_params[:, 1] + 15, average_params[:, 0], s=40, c=best_assigns.iloc[0, :], cmap=color_palette, edgecolors='k', alpha=alphas)
        plt.xlabel('Phasicity')
        plt.ylabel('Jump')
        plt.title('FEU Space')
        cbar = plt.colorbar()
        cbar.set_label('Cluster')
        plt.tight_layout()
        plt.savefig(f'{title}/{title}_params.png', dpi = 500)
        plt.close()
        print(f'Saving scatter plot of cluster parameters to {title}/{title}_params.png')

    return best_assigns, best_params

def plot_raster(raster, x_axis=None, ax=None, ms=10, offset=0):
    plt.grid('off')
    n_trials, trial_len = raster.shape
    if x_axis is None:
        x_axis = np.arange(trial_len)
    for i in range(n_trials):
        mask = (raster[i] > 0)
        if ax:
            ax.plot(x_axis[mask], (i+1+offset) * np.ones(mask.sum()), 'k.', markersize=ms)
        else:
            plt.plot(x_axis[mask], (i+1+offset) * np.ones(mask.sum()), 'k.', markersize=ms)

def make_raster_fig(data, t_stimulus, best_assigns, title):
    data_shape = data.values[0][0].shape
    print(f'Data shape: {data_shape}')
    x_axis = np.arange(-t_stimulus, data_shape[1]-t_stimulus)

    unique_clusters = np.sort(np.unique(best_assigns.iloc[0, :]))
    n_clusters = len(unique_clusters)
    n_cols = 3
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, k in enumerate(unique_clusters):
        ax = axes[idx]
        series = best_assigns.iloc[0, 0:]
        ensemble = series[series == k]
        ensemble_ids = ensemble.index.astype(int)
        subset = data.iloc[ensemble_ids]
        for raster in subset["data"]:
            plot_raster(raster, ms=1, x_axis=x_axis, ax=ax)
        ax.axvline(0, c='r')
        ax.set_title(f'Cluster {k}, Neuron Count: {len(subset)}')

    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(f'{title}/{title}_raster_clusters.png')
    # plt.show()



# Make rasters indepedently, 1 image for each cluster
# def make_raster_fig(data, t_stimulus, best_assigns, title):

#     data_shape = data.values[0][0].shape
#     print(f'Data shape: {data_shape}')
#     x_axis = np.arange(-t_stimulus, data_shape[1]-t_stimulus)

#     for k in np.sort(np.unique(best_assigns.iloc[0, :])):
#         plt.figure()

#         series = best_assigns.iloc[0, 1:]
#         ensemble = series[series == k]
#         ensemble_ids = ensemble.index.astype(int)
#         subset = data.iloc[ensemble_ids]
#         for raster in subset["data"]:
#             plot_raster(raster, ms=1, x_axis=x_axis)
#         plt.axvline(0, c='r')
#         plt.title(f'Cluster {k + 1}, Neuron Count: {len(subset)}')
#         plt.savefig(f'{title}/{title}_raster_cluster_{k+1}.png')
#         # plt.show()
