from Training import trainH

"""
main script

"""

# data_dir: the directory in which datasets are stored
DATA_DIR = ''
EPOCHS = 1000
STEPS = 1 # input/output size
batch_size = 24000 # batch size
LR = 3e-4 # initial learning rate
INFERENCE = False
MODEL_SAVE_DIR = '' # Models folder


MODEL_NAME = 'Chord_finals' #### Model name


trainH(data_dir=DATA_DIR,
      save_folder=MODEL_NAME,
      model_save_dir=MODEL_SAVE_DIR,
      learning_rate=LR,
      epochs=EPOCHS,
      batch_size=batch_size,
      num_steps=STEPS,
      inference=INFERENCE)
