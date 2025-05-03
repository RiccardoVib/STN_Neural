# Sine, Trasnient, Noise Neural Modeling of Piano Notes

This code repository for the article _Sine, Trasnient, Noise Neural Modeling of Piano Notes_, Frontiers in Signal Processing 4 (2025).

This repository contains all the necessary utilities to use our architecture. Find the code located inside the "./Code" folder, and the weights of pre-trained models inside the "./Weights" folder

Visit our [companion page with audio examples](https://riccardovib.github.io/STN_Neural_pages/)


### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)

<br/>

# Datasets
Datsets is available at the following link:
[Piano Recordings](https://www.kaggle.com/datasets/riccardosimionato/pianorecordingssinglenotes)

# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./Code
pip install -r requirements.txt
```


The piano notes generation is split into three component: Harmonic, Transient and Noise.

#### Harmonic Module

To train harmonic models,
```
cd ./Code/HarmonicModule
```

and use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. Available: [DatasetSingleNote_split_, DatasetSingleNoteGrand_split_] [str] (default=" ")
* --epochs - Number of training epochs. [int] (defaut=60)
* --batch_size - The size of each batch [int] (default=2**17)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --harmonics - Number of harmonics to synthetize [int] (default=24)
* --phantom - If include phantom partials [bool] (default=True)
* --phase = which phase to train: 'A' train partials amplitudes, 'B' the inharmonic coefficient [str] (default='A')
* --keys = which key model to train: [[str]] (default=['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'])
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)

Example training case: 
```
cd ./Code/

python starter.py --dataset 'DatasetSingleNote_split' --harmonics 24 --phase 'A' --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./Code/
python starter.py --dataset 'DatasetSingleNote_split' --harmonics 24 --only_inference True
```

#### Transient Module

To train transient models,
```
cd ./Code/TransientModule
```

and use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. Available: [DatasetSingleNote_split_, DatasetSingleNoteGrand_split_]. [str] (default=" ")
* --epochs - Number of training epochs. [int] (defaut=60)
* --batch_size - The size of each batch [int] (default=60)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --keys = which key model to train: [[str]] (default=['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'])
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)

Example training case: 
```
cd ./Code/

python starter.py --dataset 'DatasetSingleNote_split' --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./Code/
python starter.py --dataset 'DatasetSingleNote_split' --only_inference True
```

#### Noise Module

To train Noise models,
```
cd ./Code/NoiseModule
```

and use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. Available: [DatasetSingleNote_split_, DatasetSingleNoteGrand_split_]. [str] (default=" ")
* --epochs - Number of training epochs. [int] (defaut=60)
* --batch_size - The size of each batch. [int] (default=60)
* --num_steps - Number of samples to generate each iteration. [int] (default=1024)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --keys = which key model to train: [[str]] (default=['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'])
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)

Example training case: 
```
cd ./Code/

python starter.py --dataset 'DatasetSingleNote_split' --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./Code/
python starter.py --dataset 'DatasetSingleNote_split' --only_inference True
```
