import pandas as pd
from helpers import *


csv_filename = 'steering.csv'
data_path = "../CND/racecar_data/"

TEST = False
if TEST:
    data_path = "../CND/racecar_data/test/"
else:
    data_path = "../CND/racecar_data/"

csv_path = data_path + csv_filename
imgs_path = data_path + '/imgs/'

data_csv_df = pd.read_csv(csv_path, index_col=False, header=None, skiprows=[0])

data_csv_df.columns = ['index', 'center', 'steer', 'throttle']
data_csv_df = data_csv_df.drop('index', axis=1)
data_csv_df['steer'] = data_csv_df['steer'].astype(float)

print(len(data_csv_df))
# randomly drop 0-steering data
data_csv_df = data_csv_df.drop(data_csv_df.query('-0.0005 < steer < 0.0005').sample(frac=.7).index)
print(len(data_csv_df))

# flip images horizontally, add to dataframe
data_csv_df['steer_flip'] = -data_csv_df['steer']

print("generating flipped data for center...")
filelist_center_img_flip = []
for filename in data_csv_df['center']:
    filename = imgs_path + filename
    # img = Image.open(filename.strip()).transpose(Image.FLIP_LEFT_RIGHT)
    flip_filename = filename.strip().rstrip('.jpg') + 'flipped.jpg'
    # img.save(flip_filename)
    flip_filename = flip_filename.lstrip(imgs_path)
    filelist_center_img_flip.append(flip_filename)

data_csv_df['center_flipped'] = pd.Series(filelist_center_img_flip).values

save_filename = csv_path.rstrip('.csv') + '_aug.csv'
data_csv_df.to_csv(save_filename, index=False)
