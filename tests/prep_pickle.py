import unittest
import numpy as np
import torch
import os
import pickle
from dpnssm.prep import prep_pickle

class TestPrepPickle(unittest.TestCase):
    def setUp(self):
        self.dst = 'test.pt'  # Destination file

    def test_prep_pickle_numpy(self):
        data = np.random.rand(5, 5)  # Create a numpy array
        with open('test.pkl', 'wb') as f:
            pickle.dump(data, f)
        prep_pickle('test.pkl', self.dst)
        loaded_data = torch.load(self.dst)
        self.assertTrue(isinstance(loaded_data[0], torch.Tensor))  # Check if the data is a tensor
        self.assertEqual(loaded_data[0].shape, torch.from_numpy(data).shape)  # Check if the shape is preserved

    def test_prep_pickle_list(self):
        data = [[1, 2, 3, 4, 5], [2, 4, 5, 6, 6]]  # Create a list
        with open('test.pkl', 'wb') as f:
            pickle.dump(data, f)
        prep_pickle('test.pkl', self.dst)
        loaded_data = torch.load(self.dst)
        self.assertTrue(isinstance(loaded_data[0], torch.Tensor))  # Check if the data is a tensor
        self.assertEqual(loaded_data[0].shape, torch.from_numpy(np.array(data)).shape)  # Check if the shape is preserved

    def tearDown(self):
        os.remove(self.dst)  # Clean up the test file
        os.remove('test.pkl')

if __name__ == '__main__':
    unittest.main()