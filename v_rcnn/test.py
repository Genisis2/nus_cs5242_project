import pandas as pd
from ast import literal_eval
import os, shutil

df = pd.read_csv('df2.csv', usecols=['filename', 'bbox_label'], converters={'bbox_label': literal_eval})
filenames = df['filename'].to_list()
bbox_labels = df['bbox_label'].to_list()