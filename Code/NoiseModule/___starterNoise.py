from TrainingNoiseModel import train


"""
main script

"""
# data_dir: the directory in which datasets are stored
DATA_DIR = ''
MODEL_SAVE_DIR = ''  # Models folder
_NAME = 'Noise_'
batch_size = 1 # batch size
INFERENCE = False
STEPS = 1024 # input/output size
LR = 3e-4  # initial learning rate

keys = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']
keys = ['F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']
keys = ['C4']
for key in keys:

      #filename = 'DatasetSingleNoteGrand_split_' + key
      filename = 'DatasetSingleNote_split_' + key
      MODEL_NAME = _NAME + filename + '_' + str(STEPS) + ''  #### Model name
      train(data_dir=DATA_DIR,
            filename=filename,
            save_folder=MODEL_NAME,
            model_save_dir=MODEL_SAVE_DIR,
            learning_rate=LR,
            batch_size=batch_size,
            num_steps=STEPS,
            inference=INFERENCE)
