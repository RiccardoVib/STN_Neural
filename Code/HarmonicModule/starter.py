from TrainingAllModel import train
import argparse


"""
main script

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Trains the harmonic piano model. Can also be used to run pure inference.')

    parser.add_argument('--model_save_dir', default='./models', type=str, nargs='?', help='Folder directory in which to store the trained models.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--dataset', default=" ", type=str, nargs='+', help='The names of the datasets to use. Available: [DatasetSingleNote_split_, DatasetSingleNoteGrand_split_]')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--batch_size', default=2**17, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--harmonics', default=24, type=int, nargs='?', help='Number of harmonics to synthetize.')

    parser.add_argument('--phase', default='A', type=str, nargs='+', help='which phase to train: A train partials amplitudes, B the inharmonic coefficient')

    parser.add_argument('--keys', default=['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'], type=[str], nargs='+', help='which key model to train')

    parser.add_argument('--phantom', default=False, type=bool, nargs='+', help='If include phantom partials.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    return parser.parse_args()


def start_train(args):
    print("######### Preparing for training/inference harmonic model #########")
    print("\n")
    for key in args.keys:
        filename = args.dataset + key
        print("Key: ", key)
        MODEL_NAME = filename + '_' + '_' + str(args.harmonics) # Model name

        if args.only_inference:
            train(data_dir=args.data_dir,
                  model_save_dir=args.model_save_dir,
                  save_folder=MODEL_NAME,
                  learning_rate=args.learning_rate,
                  epochs=args.epochs,
                  phan=args.phantom,
                  harmonics=args.harmonics,
                  phase=args.phase,
                  inference=True)

        else:
            train(data_dir=args.data_dir,
                  model_save_dir=args.model_save_dir,
                  save_folder=MODEL_NAME,
                  learning_rate=args.learning_rate,
                  epochs=args.epochs,
                  phan=args.phantom,
                  harmonics=args.harmonics,
                  phase=args.phase,
                  inference=False)


def main():
    args = parse_args()
    start_train(args)

if __name__ == '__main__':
    main()