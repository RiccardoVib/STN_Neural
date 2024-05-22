from TrainingNoiseModel import train

DATA_DIR = '../../Files/PianoSingleNoteData' #### Dataset folder
MODEL_SAVE_DIR = '../../TrainedModels' #### Models folder
MODEL_NAME = 'Noise_24000IR' #### Model name

###next ir_size
INFERENCE = False
STEPS = 70000#15
LR = 3e-4
batch_size=1#162000//STEPS#8
#140000
train(data_dir=DATA_DIR,
      save_folder=MODEL_NAME,
      model_save_dir=MODEL_SAVE_DIR,
      learning_rate=LR,
      batch_size=batch_size,
      num_steps=STEPS,
      inference=INFERENCE)
