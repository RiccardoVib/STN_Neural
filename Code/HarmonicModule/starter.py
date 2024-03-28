from TrainingModel import train

DATA_DIR = '../../../Files/PianoSingleNoteData'  #### Dataset folder
MODEL_SAVE_DIR = '../../../TrainedModels'  #### Models folder
MODEL_NAME = 'Test' #### Model name
INFERENCE = False
HARMONICS = 24
STEPS = 60
LR = 3e-4


PHASE = 'B'

train(data_dir=DATA_DIR,
      save_folder=MODEL_NAME,
      model_save_dir=MODEL_SAVE_DIR,
      learning_rate=LR,
      num_steps=STEPS,
      harmonics=HARMONICS,
      phase=PHASE,
      inference=INFERENCE)

print("---------Finish B---------")
print('\n')

PHASE = 'A'

train(data_dir=DATA_DIR,
      save_folder=MODEL_NAME,
      model_save_dir=MODEL_SAVE_DIR,
      learning_rate=LR,
      num_steps=STEPS,
      harmonics=HARMONICS,
      phase=PHASE,
      inference=INFERENCE)

print("---------Finish A---------")
print('\n')
