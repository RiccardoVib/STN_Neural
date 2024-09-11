from TrainingNoiseModel import train

DATA_DIR = '../../Files/PianoSingleNoteDataGrand' #### Dataset folder
DATA_DIR = '../../Files/PianoSingleNoteData' #### Dataset folder
MODEL_SAVE_DIR = '../../TrainedModels'  #### Models folder
_NAME = 'Noise_' #### Model name

###next ir_size
INFERENCE = False
STEPS = 1024#24000#15
LR = 3e-4
batch_size=1#262144//1024#1
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
