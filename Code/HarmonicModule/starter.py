from TrainingAllModel import train

DATA_DIR = '../../Files/PianoSingleNoteData/'  #### Dataset folder
#DATA_DIR = '../../Files/PianoSingleNoteDataGrand/'
MODEL_SAVE_DIR = '../../TrainedModels'  #### Models folder
INFERENCE = False
HARMONICS = 24
STEPS = 1
LR = 3e-4
g = 9
phan = False
A = True
B = True
    
if INFERENCE:
    B = False

keys = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']

batch_size = 2**17
minibatch_size = 1

for key in keys:

    filename = 'DatasetSingleNote_split_' + key
    #filename = 'DatasetSingleNoteGrand_split_' + key

    MODEL_NAME = filename + '_' + str(STEPS) + '_' + str(HARMONICS) + ''#### Model name

        
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
              minibatch_size=minibatch_size,
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
              minibatch_size=minibatch_size,
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
              minibatch_size=minibatch_size,
              harmonics=HARMONICS,
              phase=PHASE,
              inference=INFERENCE)

        print("---------Finish A---------")
        print('\n')
