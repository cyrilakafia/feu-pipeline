import unittest
import pandas as pd
import numpy as np
import torch
import os
from feu.prep import prep_csv

class TestPrepFunctions(unittest.TestCase):
    def setUp(self):
        self.dst = 'test.pt'  # Destination file

    def test_prep_csv(self):
        df = pd.DataFrame(np.random.rand(5, 5))  # Create a DataFrame
        df.to_csv('test.csv', index=False)  # Save the DataFrame to a CSV file
        prep_csv('test.csv', self.dst)
        loaded_data = torch.load(self.dst)
        self.assertTrue(isinstance(loaded_data[0], torch.Tensor))  # Check if the data is a tensor
        os.remove('test.csv')  # Clean up the test file

    def tearDown(self):
        os.remove(self.dst)  # Clean up the test file

if __name__ == '__main__':
    unittest.main()