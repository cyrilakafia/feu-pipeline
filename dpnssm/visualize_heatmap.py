import matplotlib.pyplot as plt
import matplotlib
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

def viz_heatmap(run, iter, max_clusters=20):
    '''
    Visualize the cluster assignments as a heatmap

    Args:
    run (str): the run number
    iter (int): the number of iterations

    
    Returns:
    best_assigns (pd.DataFrame): the best cluster assignments
    best_params (pd.DataFrame): the best cluster parameters
    
    Saves best cluster assignments to outputs/sim{run}_best_assigns.csv
    Saves best cluster parameters to outputs/sim{run}_best_params.csv
    Saves a scatter plot of cluster parameters to outputs/sim{run}_params.png
    Saves a heatmap of the cluster assignments to outputs/sim{run}_assigns.png
    '''

    if iter < 510:
        assigns = pd.read_csv(f"outputs/sim{run}_assigns.csv", header=None)
        params = pd.read_csv(f"outputs/sim{run}_params.tsv", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters))

    else:
        assigns = pd.read_csv(f"outputs/sim{run}_assigns.csv", header=None).iloc[500:].reset_index(drop=True)
        params = pd.read_csv(f"outputs/sim{run}_params.tsv", header=None, sep='\t', on_bad_lines='warn', names=np.arange(0, max_clusters)).iloc[500:].reset_index(drop=True)

    params.fillna('0,0', inplace=True)
    split_params = params.applymap(lambda x: np.array(x.split(','), dtype=float))

    average_params = []

    for count, datapoint in enumerate(assigns):
        datapoint_params = []
        print(f'datapoint {assigns[datapoint]}')
        for index, assignments in enumerate(assigns[datapoint]):
            print(f'index: {index}')
            print(f'assignments: {split_params[assignments]}')
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
    best_params.to_csv(f'outputs/sim{run}_best_params.csv')
    print(f'Found best parameters. Saving to file to outputs/sim{run}_best_params.csv')

    # rotate the df
    best_assigns = best_assigns.T

    # save best_assigns to csv file
    best_assigns.to_csv(f'outputs/sim{run}_best_assigns.csv')
    print(f'Found best assigns. Saving to file to outputs/sim{run}_best_assigns.csv')

    reorder = assigns.loc[min_dist_idx].sort_values().index

    # plot cluster assignments
    plt.imshow(mean_coo[np.ix_(reorder, reorder)], cmap="coolwarm")
    plt.colorbar()

    N = len(assigns.columns)

    plt.yticks(np.arange(N), reorder, fontsize=5)
    plt.xticks(np.arange(N), reorder, fontsize=5, rotation=90)
    plt.tight_layout()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists(f'outputs/sim{run}_assigns.png'):
        plt.savefig(f'outputs/sim{run}_assigns.png', dpi = 500)
    else:
        oldfile = f'sim{run}_assigns.png'
        os.remove(f'outputs/{oldfile}')
        plt.savefig(f'outputs/sim{run}_assigns.png', dpi = 500)
        
    plt.close()
    print(f'Saving cluster assignments heatmap to outputs/sim{run}_assigns.png')

    # make params figure
    plt.scatter(average_params[:, 1] + 15, average_params[:, 0], s=20)
    plt.xlabel('phasicity')
    plt.ylabel('jump')
    plt.title('FEU Space')
    plt.tight_layout()
    plt.savefig(f'outputs/sim{run}_params.png', dpi = 500)
    plt.close()
    print(f'Saving scatter plot of cluster parameters to outputs/sim{run}_params.png')


    return best_assigns, best_params

