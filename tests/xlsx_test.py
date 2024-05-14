import unittest
import pandas as pd
import numpy as np
import torch
import os
from feu.prep import prep_xlsx

class TestPrepXlsx(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(np.random.rand(10, 10))  # Create a DataFrame
        self.dst = 'test.pt'  # Destination file
        self.data.to_excel('test.xlsx',index=False)  # Save the DataFrame to an Excel file

    def test_prep_xlsx(self):
        prep_xlsx('test.xlsx', self.dst)
        loaded_data = torch.load(self.dst)
        self.assertTrue(isinstance(loaded_data[0], torch.Tensor))  # Check if the data is a tensor
        self.assertEqual(loaded_data[0].shape, torch.from_numpy(self.data.to_numpy()).shape)  # Check if the shape is preserved

    def tearDown(self):
        os.remove(self.dst)  # Clean up the test file
        os.remove('test.xlsx')

if __name__ == '__main__':
    unittest.main()