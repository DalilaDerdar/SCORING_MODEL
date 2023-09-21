import logging
logging.getLogger('mlflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
import sys
sys.path.insert(0, 'notebook/Streamlit_pages')
import unittest
import numpy as np
from modelisation import custom_cost_only_fn

class TestCustomCostOnlyFn(unittest.TestCase):
    def test_return_type(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        result = custom_cost_only_fn(y_true, y_pred)
        self.assertIsInstance(result, float)
        print("test_return_type passed")
        
    def test_positive_profit(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 0, 1, 1, 1, 1]  # 0 false negative, 8 true predictions
        self.assertGreaterEqual(custom_cost_only_fn(y_true, y_pred), 0)
        print("test_positive_profit passed")
       
    def test_negative_profit(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [1, 1, 1, 1, 0, 0, 0, 0]  # 4 false negatives, 4 false positives
        self.assertLessEqual(custom_cost_only_fn(y_true, y_pred), 0)
        print("test_negative_profit passed")
       
if __name__ == '__main__':
    unittest.main()
