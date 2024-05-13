import torch
import numpy as np
import pickle
import pandas as pd
import pynwb

torch.set_default_dtype(torch.double)

# check the file type
def check_file_type(data):
    """
    Check the file type of the data

    Args:
    data: str, path to the data file

    Returns:
    str: file type
    """
    if data.endswith('.pkl') or data.endswith('.pickle'):
        return 'pickle'
    elif data.endswith('.npy'):
        return 'numpy'
    elif data.endswith('.csv'):
        return 'csv'
    elif data.endswith('.xls') or data.endswith('.xlsx'):
        return 'xls'
    elif data.endswith('.txt'):
        return 'txt'
    elif data.endswith('.nwb'):
        return 'nwb'
    elif data.endswith('.p'):
        return 'torch'
    elif data.endswith('.pt'):
        return 'torch'
    else:
        return 'Unsupported file type'

def prep_pickle(data, dst):
    """
    Preprocess the pickle data
    
    Args:
    data: str, path to the data file
    dst: str, path to save the preprocessed data

    Returns:
    None

    Saves the preprocessed data to the specified destination
    """
    with open(f'{data}', 'rb') as f:
        x = pickle.load(f)
    try:
        x_tensor = torch.from_numpy(x)
        x_tensor = (x_tensor, )
    except TypeError:
        x_tensor = np.array(x)                  # convert to numpy in the case of simple lists 
        x_tensor = torch.from_numpy(x_tensor)
        x_tensor = (x_tensor, )

    torch.save(x_tensor, dst)

    print('Preprocessing done')

def prep_numpy(data, dst):
    """
    Preprocess the numpy data

    Args:
    data: str, path to the data file
    dst: str, path to save the preprocessed data

    Returns:
    None

    Saves the preprocessed data to the specified destination
    """
    x = np.load(data)
    x_tensor = torch.from_numpy(x)

    x_tensor = (x_tensor, )
    torch.save(x_tensor, dst)

    print('Preprocessing done')

def prep_torch(data, dst):
    x = torch.load(data)
    
    # check if the data is a tuple
    if isinstance(x, tuple):
        pass
    else:
        x = (x, )

    # check if the data is a tensor
    if not isinstance(x[0], torch.Tensor):
        x_tensor = (torch.tensor(x[0]), )

    else:
        x_tensor = x

    # check to see if the data is 2D
    if len(x_tensor[0].shape) != 2:
        return 'Data must be 2D'
    
    torch.save(x_tensor, dst)

    print('Preprocessing done')

def prep_csv(data, dst):
    df = pd.read_csv(data)
    x = df.to_numpy()
    x_tensor = torch.from_numpy(x)
    x_tensor = (x_tensor, )
    torch.save(x_tensor, dst)

    print('Preprocessing done')

def prep_xlsx(data, dst):
    df = pd.read_excel(data)
    x = df.to_numpy()
    x_tensor = torch.from_numpy(x)
    x_tensor = (x_tensor, )
    torch.save(x_tensor, dst)

    print('Preprocessing done')

def prep_txt(data, dst):
    x = np.loadtxt(data)
    x_tensor = torch.from_numpy(x)
    x_tensor = (x_tensor, )
    torch.save(x_tensor, dst)

def prep_nwb(data, dst):
    pass


def prep(data, dst):
    """
    Preprocess the data

    Args:
    data: str, path to the data file
    dst: str, path to save the preprocessed data

    Returns:
    None

    Saves the preprocessed data to the specified destination
    """
    file_type = check_file_type(data)
    if file_type == 'pickle':
        prep_pickle(data, dst)
    elif file_type == 'numpy':
        prep_numpy(data, dst)
    elif file_type == 'csv':
        prep_csv(data, dst)
    elif file_type == 'xls':
        prep_xlsx(data, dst)
    elif file_type == 'txt':
        prep_txt(data, dst)
    elif file_type == 'torch':
        prep_torch(data, dst)
    else:
        return 'Unsupported file type'

if __name__ == '__main__':
    prep_pickle("../test_data/original_pickle_data.pickle", "../test_data/pickle_data_after_prep.p")
    print('Test 1 pass')

    prep_pickle("../test_data/array_30_200.pkl", "../test_data/array_30_200_after_prep.p")
    print('Test 2 pass')

    prep_numpy('../test_data/random_numpy_data.npy', '../test_data/numpy_data_after_prep.p')
    print('Test 3 pass')