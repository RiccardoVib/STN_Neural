from TrainingTransientModel import train

"""
main script

"""

# data_dir: the directory in which datasets are stored
DATA_DIR = ''
epochs = 200
batch_size = 60 # batch size
LR = 3e-4 # initial learning rate
INFERENCE = False
MODEL_SAVE_DIR = '../../TrainedModels' #### Models folder


keys = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']

for key in keys:

      filename = 'DatasetSingleNoteGrand_split_' + key

      MODEL_NAME = 'T_Grand_' + filename#### Model name

      train(data_dir=DATA_DIR,
            save_folder=MODEL_NAME,
            model_save_dir=MODEL_SAVE_DIR,
            learning_rate=LR,
            filename=filename,
            batch_size=batch_size,
            inference=INFERENCE)