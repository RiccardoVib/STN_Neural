from TrainingNoiseModel import train

DATA_DIR = '../../../Files/PianoSingleNoteData' #### Dataset folder
MODEL_SAVE_DIR = '../../../TrainedModels' #### Models folder
MODEL_NAME = 'Test' #### Model name
# poi coeff di freq
INFERENCE = False
STEPS = 120
LR = 3e-4

train(data_dir=DATA_DIR,
      save_folder=MODEL_NAME,
      model_save_dir=MODEL_SAVE_DIR,
      learning_rate=LR,
      num_steps=STEPS,
      inference=INFERENCE)