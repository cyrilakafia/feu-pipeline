import unittest
from feu.prep import check_file_type

class TestCheckFileType(unittest.TestCase):
    def test_check_file_type(self):
        self.assertEqual(check_file_type('data.pkl'), 'pickle')
        self.assertEqual(check_file_type('data.pickle'), 'pickle')
        self.assertEqual(check_file_type('data.npy'), 'numpy')
        self.assertEqual(check_file_type('data.txt'), 'Unsupported file type')
        self.assertEqual(check_file_type('data.csv'), 'csv')
        self.assertEqual(check_file_type('data.xls'), 'xls')
        self.assertEqual(check_file_type('data.xlsx'), 'xls')
        self.assertEqual(check_file_type('data.nwb'), 'nwb')

if __name__ == '__main__':
    unittest.main()