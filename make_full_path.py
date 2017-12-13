import pandas as pd
from helpers import *

csv_filename = 'steering_aug.csv'
data_path = "../CND/racecar_data/"

img_path = "/home/boliu1/Projects/CND/racecar_data/imgs/crop/"

csv_path = data_path + csv_filename
print(csv_path)

data_csv_df = pd.read_csv(csv_path, index_col=False, header=None, skiprows=[0])
data_csv_df.columns = ['center', 'steer', 'throttle', 'steer_flip', 'center_flipped']

data_csv_df['center'] = img_path + data_csv_df['center']
data_csv_df['center_flipped'] = img_path + data_csv_df['center_flipped']

data_csv_df.to_csv('../CND/racecar_data/steering_aug_fullpath.csv')