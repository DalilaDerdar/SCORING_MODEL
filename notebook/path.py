
from pathlib import Path
import os

path_to_sampledf = os.path.join(os.getcwd(), 'data', 'sampledf.csv')

if os.getcwd() == 'scoring_model':
    path_to_data_folder = Path(os.getcwd(), 'data')
elif os.getcwd() == 'notebook':
    path_to_data_folder = Path(os.getcwd(), '..', 'data')
else:
    raise NotImplementedError('Please run this notebook from the scoring_model or notebook folder, the current working directory is '+os.getcwd())

