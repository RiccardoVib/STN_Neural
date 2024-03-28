from TrainingTransientModel import train

DATA_DIR = '../../../Files/PianoSingleNoteData' #### Dataset folder
MODEL_SAVE_DIR = '../../../TrainedModels' #### Models folder
MODEL_NAME = 'Test' #### Model name

INFERENCE = False
LR = 3e-4

train(data_dir=DATA_DIR,
      save_folder=MODEL_NAME,
      model_save_dir=MODEL_SAVE_DIR,
      learning_rate=LR,
      inference=INFERENCE)