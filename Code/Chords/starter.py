from Training import train
import argparse


"""
main script

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Trains the chord piano model. Can also be used to run pure inference.')

    parser.add_argument('--model_save_dir', default='./models', type=str, nargs='?', help='Folder directory in which to store the trained models.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--dataset', default=" ", type=str, nargs='+', help='The names of the datasets to use. Available: DiskChordUpright_split')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--batch_size', default=8, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--mini_batch_size', default=2400, type=int, nargs='?', help='Mini Batch size.')

    parser.add_argument('--steps', default=1, type=int, nargs='?', help='Number of samples to generate each iteration.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    return parser.parse_args()


def start_train(args):
    print("######### Preparing for training/inference transient model#########")
    print("\n")

    train(data_dir=args.data_dir,
        model_save_dir=args.model_save_dir,
        save_folder=f'{args.dataset}_{args.steps}',
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        inference=args.only_inference)

def main():
    args = parse_args()
    start_train(args)

if __name__ == '__main__':
    main()