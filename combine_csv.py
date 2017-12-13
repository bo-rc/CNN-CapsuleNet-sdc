import pandas as pd
import glob
from helpers import *


csv_filename = 'driving_log.csv'
data_path = "../CND/racecar_data/"
filenames = glob.glob(data_path + "/*.csv")

img_path = "../CND/racecar_data/imgs/"

data_test_path = "../CND/racecar_data/test/"

TEST = True
if TEST:
    data_csv_path = data_test_path + csv_filename
else:
    data_csv_path = data_path + csv_filename

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, index_col=False, header=None, skiprows=[0]))

data_csv_df = pd.DataFrame(pd.concat(dfs))

data_csv_df.to_csv("steering.csv")