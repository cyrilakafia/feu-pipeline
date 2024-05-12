import unittest
import torch
import os
from dpnssm.prep import prep_torch

class TestPrepTorch(unittest.TestCase):
    def setUp(self):
        self.data = torch.rand(5, 5)  # Create a 2D tensor
        torch.save(self.data, 'odata.p')
        self.orig = 'odata.p'
        self.dst = 'pdata.p'  # Destination file

    def test_prep_torch(self):
        prep_torch(self.orig, self.dst)
        loaded_data = torch.load(self.dst)
        self.assertTrue(isinstance(loaded_data[0], torch.Tensor))  # Check if the data is a tensor
        self.assertEqual(loaded_data[0].shape, self.data.shape)  # Check if the shape is preserved

    def tearDown(self):
        os.remove(self.orig)  # Clean up the test file
        os.remove(self.dst)  # Clean up the test file

if __name__ == '__main__':
    unittest.main()