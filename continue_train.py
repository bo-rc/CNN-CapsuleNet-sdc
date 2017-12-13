import keras
from keras import optimizers
import pandas as pd
from helpers import *

# Load the model
model_file = './keras_nvidiaNet.h5'
model = keras.models.load_model(model_file)

BATCH_SIZE = 64*2
BATCH_SIZE_VALID = 32*2
EPOCHS = 3

print(model.summary())

data_csv_path = '../CND/racecar_data/steering_aug_fullpath.csv'
data_path = "../CND/racecar_data/imgs/crop/"

data_csv_df = pd.read_csv(data_csv_path, index_col=False)

data = data_csv_df.sample(n=len(data_csv_df))

split_train = int(0.75*len(data))
split_valid = int(0.95*len(data))
split_test = len(data)

data_train = data[:split_train]
data_valid = data[split_train:split_valid]
data_test = data[split_valid:split_test]

n_data_train = len(data_train)
n_steps = int(n_data_train/BATCH_SIZE)

n_data_valid = len(data_valid)
n_steps_valid = int(n_data_valid/BATCH_SIZE_VALID)

model.compile(optimizer=optimizers.Adam(lr=0.0005), loss='mse')

values = model.fit_generator(train_sample_generator(data_train, BATCH_SIZE),
                             validation_data=valid_sample_generator(data_valid, BATCH_SIZE_VALID),
                             steps_per_epoch=n_steps,
                             validation_steps=n_steps_valid,
                             epochs=EPOCHS)

new_file = model_file.rstrip('.h5') + '_continue.h5'
model.save(new_file)

