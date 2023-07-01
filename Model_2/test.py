import os
import torch
import torch.optim as optim
import argparse
import pandas as pd 

from preprocess import preprocess
from train_utils import train, generate
from model import EncoderModel, LSTMEncoderModel
from utils import ArgumentParser, load_data, get_vocabulary

def main(args,char_to_idx,idx_to_char):

    if args.model == 'dense':
        model = EncoderModel(context_length = args.context_length, device = args.device)
    elif args.model == 'rnn':
        model = LSTMEncoderModel(vocab_size=len(char_to_idx))
    else:
        raise ValueError
    
    model.to(args.device)

    # load the model
    model.load_state_dict(torch.load(f'models/model_{args.model}_{args.context_length}.pt'))

    # generate text
    preds = generate(model, char_to_idx, idx_to_char, args, args.prompt)
    print()
    print()
    print(args.prompt+preds)

if __name__ == '__main__':

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model', type=str, default='rnn')
    parser.add_argument('--context_length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--prompt',type = str)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set the device based on the GPU availability and GPU ID argument
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')


    # Load char_to_idx dictionary from CSV
    char_to_idx_df = pd.read_csv('char_to_idx.csv')
    char_to_idx = dict(zip(char_to_idx_df['char'], char_to_idx_df['idx']))

    # Load idx_to_char dictionary from CSV
    idx_to_char_df = pd.read_csv('idx_to_char.csv')
    idx_to_char = dict(zip(idx_to_char_df['idx'], idx_to_char_df['char']))

    main(args,char_to_idx,idx_to_char)
