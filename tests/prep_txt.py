import unittest
import pandas as pd
import numpy as np
import torch
import os
from dpnssm.prep import prep_txt

class TestPrepFunctions(unittest.TestCase):
    def setUp(self):
        self.dst = 'test.pt'  # Destination file

    def test_prep_txt(self):
        data = np.random.rand(5, 5)  # Create a numpy array
        np.savetxt('test.txt', data)  # Save the numpy array to a text file
        prep_txt('test.txt', self.dst)
        loaded_data = torch.load(self.dst)
        self.assertTrue(isinstance(loaded_data[0], torch.Tensor))  # Check if the data is a tensor
        os.remove('test.txt')  # Clean up the test file

    def tearDown(self):
        os.remove(self.dst)  # Clean up the test file

if __name__ == '__main__':
    unittest.main()