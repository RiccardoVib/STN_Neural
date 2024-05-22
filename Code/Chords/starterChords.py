from Training import trainH

DATA_DIR = '../../Files/PianoChordNoteData' #### Dataset folder
MODEL_SAVE_DIR = '../../TrainedModels' #### Models folder

INFERENCE = False
LR = 3e-4
STEPS = 1
batch_size =60

MODEL_NAME = 'name' #### Model name
EPOCHS = 1000
trainH(data_dir=DATA_DIR,
      save_folder=MODEL_NAME,
      model_save_dir=MODEL_SAVE_DIR,
      learning_rate=LR,
      epochs=EPOCHS,
      batch_size=batch_size,
      num_steps=STEPS,
      inference=INFERENCE)
