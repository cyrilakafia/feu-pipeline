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

def find_best_clust_and_params(title, output_folder, assigns_df = None, params_df = None, max_clusters=20, figures=True):
    '''
    Visualize the cluster assignments as a heatmap

    Args:
    title (str): the title number
    assigns (pd.DataFrame): the cluster assignments
    params (pd.DataFrame): the cluster parameters
    max_clusters (int): the maximum number of clusters

    
    Returns:
    best_assigns (pd.DataFrame): the best cluster assignments
    best_params (pd.DataFrame): the best cluster parameters
    
    Saves best cluster assignments to {output_folder}_best_assigns.csv
    Saves best cluster parameters to {output_folder}_best_params.csv
    Saves a scatter plot of cluster parameters to {output_folder}_params.png
    Saves a heatmap of the cluster assignments to {output_folder}_assigns.png
    '''

    if assigns_df is None or params_df is None:
        assigns = pd.read_csv(f"{output_folder}_assigns.csv", header=None)
    else:
        assigns = pd.read_csv(f"{assigns_df}", header=None)

    iter = assigns.shape[0]

    if iter < 510:
        if assigns_df is None or params_df is None:
            params = pd.read_csv(f"{output_folder}_params.tsv", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters))
        else:
            params = pd.read_csv(f"{params_df}", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters))

    else:
        if assigns_df is None or params_df is None:
            assigns = pd.read_csv(f"{output_folder}_assigns.csv", header=None).iloc[500:].reset_index(drop=True)
            params = pd.read_csv(f"{output_folder}_params.tsv", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters)).iloc[500:].reset_index(drop=True)
        else:
            assigns = pd.read_csv(f"{assigns_df}", header=None).iloc[500:].reset_index(drop=True)
            params = pd.read_csv(f"{params_df}", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters)).iloc[500:].reset_index(drop=True)
        

    params.fillna('0,0', inplace=True)
    split_params = params.applymap(lambda x: np.array(x.split(','), dtype=float))

    coos = np.stack([ids_to_coo(x) for _, x in assigns.iterrows()])
    mean_coo = coos.mean(axis=0)
    dists = np.linalg.norm(coos - mean_coo, axis=(1, 2))   # dists measures how different each clustering assignment is from the average assignment
    min_dist_idx = np.argmin(dists)     # identifies the most representative (closest to average) clustering assignment
    match_idxs = np.where(np.all(coos == coos[min_dist_idx], axis=(1, 2)))[0]       # match_idxs lists all samples that have the exact same clustering assignment as the most representative one
    print(f"Found {len(match_idxs)} samples out of {len(assigns)} that match the best assignment.")
    
    # reorder sample ids to match best assignment
    reorder = assigns.loc[min_dist_idx].sort_values().index
    
    if figures:
        # Plot cluster assignments
        plt.imshow(mean_coo[np.ix_(reorder, reorder)], cmap="coolwarm")
        plt.colorbar()

        # Add chosen clustering in outline
        counts = np.bincount(assigns.loc[min_dist_idx])
        N = len(assigns.columns)
        start = -0.5
        for c in counts:
            rect = patches.Rectangle((start, start), c, c, linewidth=1, edgecolor='lime', facecolor='none')
            start += c
            plt.gca().add_patch(rect)

        plt.yticks(np.arange(N), reorder, fontsize=5)
        plt.xticks(np.arange(N), reorder, fontsize=5, rotation=90)

        if not os.path.exists(f'{title}'):
            os.makedirs(f'{title}')
        if not os.path.exists(f'{output_folder}_assigns.png'):
            plt.savefig(f'{output_folder}_assigns.png', dpi = 500)
        else:
            oldfile = f'{title}_assigns.png'
            os.remove(f'{title}/{oldfile}')
            plt.savefig(f'{output_folder}_assigns.png', dpi = 500)
            
        plt.close()
        print(f'Saving cluster assignments heatmap to {output_folder}_assigns.png')


    ######### Single Data Point (Neuron) Figures ############

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

    best_assigns = pd.DataFrame(assigns.loc[min_dist_idx])
    best_assigns.columns = ['cluster']

    if figures:
        # make params figure
        plt.figure()

        min_alpha = 0.7
        max_alpha = 0.9

        unique_clusters = np.unique(best_assigns.iloc[:, 0])
        num_clusters = len(unique_clusters)

        # Create a fixed color map for the clusters
        colors_list = [
            'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
            'magenta', 'yellow', 'teal', 'navy', 'maroon', 'lime', 'gold', 'silver', 'coral', 'lavender',
            'indigo', 'turquoise', 'salmon', 'plum', 'tan', 'khaki', 'beige', 'crimson', 'darkgreen', 'lightblue'
            ]
        
        cluster_labels = sorted(unique_clusters)
        cluster_colors = {cluster_label: colors_list[i % len(colors_list)] for i, cluster_label in enumerate(cluster_labels)}

        # Generate alphas based on cluster size or range
        if num_clusters == 1:
            alphas = np.full(len(best_assigns), max_alpha)

        else:
            alphas = min_alpha + (best_assigns.iloc[:, 0] - unique_clusters.min()) * (max_alpha - min_alpha) / (num_clusters - 1)

        for i in range(len(average_params)):
            cluster = best_assigns.iloc[i, 0]
            plt.scatter(
                average_params[i][1] + 15,
                average_params[i][0],
                s=40,
                c=[cluster_colors[cluster]],
                edgecolors='k',
                alpha=alphas[i]
            )

        # Create a custom legend for clusters
        for cluster, color in cluster_colors.items():
            plt.scatter([], [], c=[color], edgecolors='k', s=40, label=f'Cluster {cluster+1}')

        plt.xlabel('Phasicity')
        plt.ylabel('Jump')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{output_folder}_params.png', dpi = 500)
        plt.close()
        print(f'Saving scatter plot of cluster parameters to {output_folder}_params.png')
       

    ######### Ensemble figure ###################

    jump_phasicity = split_params.iloc[match_idxs]
    
    jump = jump_phasicity.applymap(lambda x: x[0])
    phasicity = jump_phasicity.applymap(lambda x: x[1])

    jump = jump.mean()
    phasicity = phasicity.mean()

    jump = pd.DataFrame([jump], index=['Mean'])
    phasicity = pd.DataFrame([phasicity], index=['Mean'])
    

    jump = jump.loc[:, (jump != 0).any(axis=0)]
    phasicity = phasicity.loc[:, (phasicity != 0).any(axis=0)]

    jump = jump.to_numpy()
    phasicity = phasicity.to_numpy()

    print(type(jump))

    if figures:
        # make params figure

        fig, ax = plt.subplots()

        ax.scatter(phasicity + 15, jump, s=counts+40)

        ax.set_xlabel("Phasicity")
        ax.set_ylabel("Jump")

        plt.savefig(f'{output_folder}_ensembles.png', dpi=500)
        plt.close()
        print(f'Saving scatter plot of cluster parameters to {output_folder}_ensembles.png')


    # Save final csv files

    # get the best assigns
    best_assigns =  pd.DataFrame(assigns.loc[min_dist_idx])
    best_assigns.columns = ['cluster id']

    # save params to file
    best_params = pd.DataFrame(average_params)
    best_params.columns = ['jump', 'phasicity']
    best_params.to_csv(f'{output_folder}_best_params.csv')
    print(f'Found best parameters. Saving to file to {output_folder}_best_params.csv')

    # rotate the df
    best_assigns = best_assigns.T

    # save best_assigns to csv file
    best_assigns.to_csv(f'{output_folder}_best_assigns.csv')
    print(f'Found best assigns. Saving to file to {output_folder}_best_assigns.csv')

    # save ensemble params to csv file
    ensemble_params = pd.DataFrame()
    ensemble_params['Jump'] = jump.tolist()[0]              # a list of list is created e.g [[1,2,3,4,5]] so indexing to pick only the inner/main list
    ensemble_params['Phasicity'] = phasicity.tolist()[0]

    ensemble_params.to_csv(f'{output_folder}_ensemble_params.csv')
    print(f'Saving ensemble params to {output_folder}_ensemble_params.csv')

    return best_assigns, best_params


#### RASTERS #####

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

def make_raster_fig(data, t_stimulus, best_assigns, title, output_folder):
    """
    data: pd.DataFrame with a column "data" containing the raster data as numpy arrays
    t_stimulus: int, the timepoint of the stimulus
    best_assigns: pd.DataFrame with a single row containing the cluster assignments
    title: str, the title of the figure
    """
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
    plt.savefig(f'{output_folder}_raster_clusters.png')
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
#         plt.savefig(f'{output_folder}_raster_cluster_{k+1}.png')
#         # plt.show()
