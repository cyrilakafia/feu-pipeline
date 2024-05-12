import unittest
import numpy as np
import torch
import os
from dpnssm.prep import prep_numpy

class TestPrepNumpy(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(5, 5)  # Create a numpy array
        self.dst = 'test.pt'  # Destination file
        np.save('test.npy', self.data)  # Save the numpy array to a file

    def test_prep_numpy(self):
        prep_numpy('test.npy', self.dst)
        loaded_data = torch.load(self.dst)
        self.assertTrue(isinstance(loaded_data[0], torch.Tensor))  # Check if the data is a tensor
        self.assertEqual(loaded_data[0].shape, torch.from_numpy(self.data).shape)  # Check if the shape is preserved

    def tearDown(self):
        os.remove(self.dst)  # Clean up the test file
        os.remove('test.npy')

if __name__ == '__main__':
    unittest.main()