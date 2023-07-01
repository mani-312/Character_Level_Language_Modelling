import os
import torch
import torch.optim as optim
import argparse
import pandas as pd

from preprocess import preprocess
from train_utils import train, generate
from model import EncoderModel, LSTMEncoderModel
from utils import ArgumentParser, load_data, get_vocabulary


def main(args: ArgumentParser) -> None:

    # data = load_data(args.raw_path)
    # data = preprocess(data)
    # with open(os.path.join(args.processed_path, 'data.txt'), 'w') as f:
    #     f.write(data)

    with open(os.path.join(args.processed_path, 'data.txt'), 'r') as f:
        data = f.read()
    # data[i] represents a character
    
    print("\n--------- Loaded Data --------\n")

    
    # get all unique characters in the corpus
    char_to_idx, idx_to_char = get_vocabulary(data)

    # Save the dict into csv

    # Save char_to_idx dictionary to CSV
    #char_to_idx_df = pd.DataFrame(char_to_idx.items(), columns=['char', 'idx'])
    #char_to_idx_df.to_csv('char_to_idx.csv', index=False)

    # Save idx_to_char dictionary to CSV
    #idx_to_char_df = pd.DataFrame(idx_to_char.items(), columns=['idx', 'char'])
    #idx_to_char_df.to_csv('idx_to_char.csv', index=False)

    vocab_size = len(char_to_idx)
    print("Number of unique characters : ",vocab_size)

    # convert the corpus into a sequence of indices
    data = [char_to_idx[ch] for ch in data]

    # create the model and optimizer
    if args.model == 'dense':
        model = EncoderModel(context_length = args.context_length, device = args.device)
    elif args.model == 'rnn':
        model = LSTMEncoderModel(vocab_size=vocab_size)
    else:
        raise ValueError
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train the model
    if args.train:
        train(model, optimizer, data, args)

    # load the model
    model.load_state_dict(torch.load(f'models/model_{args.model}_{args.context_length}.pt'))

    # generate text
    prompt = 'Help will always be given at Hogwarts to those who'
    preds = generate(model, char_to_idx, idx_to_char, args, prompt)
    print(prompt+preds)


if __name__ == '__main__':
    # args = ArgumentParser({
    #     'gpu_id': 0,
    #     'raw_path': 'dataset',
    #     'model': 'rnn',  # 'dense', 'rnn'
    #     'train': False,
    #     'processed_path': 'data',
    #     'context_length': 100,  # dense: {5, 10}, rnn: {100}
    #     'batch_size': 256,
    #     'lr': 1e-4,
    #     'num_iterations': 500_000,
    #     'log_freq': 100,
    #     'temperature': 1.0,
    # })
    # args.device = torch.device(
    #     f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--raw_path', type=str, default='dataset')
    parser.add_argument('--model', type=str, default='rnn')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--processed_path', type=str, default='data')
    parser.add_argument('--context_length', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_iterations', type=int, default=500000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set the device based on the GPU availability and GPU ID argument
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Use the parsed arguments in your code
    # ...
    # Your code here
    # ...

    main(args)
