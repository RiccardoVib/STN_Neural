from TrainingAllModel import train


"""
main script

"""
# data_dir: the directory in which datasets are stored
DATA_DIR = ''
MODEL_SAVE_DIR = '../../TrainedModels'  # Models folder
INFERENCE = False
HARMONICS = 24 # number of partials to generate
STEPS = 1 # output size
LR = 3e-4 # initial leanring rate
g = 9 # starting mode for compute the longitudinal displacements
phan = False # if include phantom partials
A = True # phase training decay rates
B = True # phase training inharmonic factor
    
if INFERENCE:
    B = False

keys = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']

batch_size = 2**17


for key in keys:

    filename = 'DatasetSingleNote_split_' + key


    MODEL_NAME = filename + '_' + str(STEPS) + '_' + str(HARMONICS) + '' # Model name

        
    if A == True:
        PHASE = 'A'

        train(data_dir=DATA_DIR,
              filename=filename,
              save_folder=MODEL_NAME,
              model_save_dir=MODEL_SAVE_DIR,
              learning_rate=LR,
              g=g,
              phan=phan,
              epochs=10000,
              batch_size=batch_size,
              num_steps=STEPS,
              harmonics=HARMONICS,
              phase=PHASE,
              inference=INFERENCE)

        print("---------Finish A---------")
        print('\n')

    if B == True:
        PHASE = 'B'
        train(data_dir=DATA_DIR,
              filename=filename,
              save_folder=MODEL_NAME,
              model_save_dir=MODEL_SAVE_DIR,
              learning_rate=LR,
              g=g,
              phan=phan,
              epochs=3000,
              batch_size=batch_size,
              num_steps=STEPS,
              harmonics=HARMONICS,
              phase=PHASE,
              inference=INFERENCE)

        print("---------Finish B---------")
        print('\n')

        
    if A == True:
        PHASE = 'A'

        train(data_dir=DATA_DIR,
              filename=filename,
              save_folder=MODEL_NAME,
              model_save_dir=MODEL_SAVE_DIR,
              learning_rate=LR,
              g=g,
              phan=phan,
              epochs=10000,
              batch_size=batch_size,
              num_steps=STEPS,
              harmonics=HARMONICS,
              phase=PHASE,
              inference=INFERENCE)

        print("---------Finish A---------")
        print('\n')
