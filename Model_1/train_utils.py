import numpy as np

import torch
import torch.nn.functional as F


def train(model, optimizer, data, args):
    model.train()
    for i in range(1, args.num_iterations + 1):
        # get the batch
        # gets random integers of size "batch_size" in the range [0,len(data) - args.context_length - 1] 
        batch_idxs = np.random.randint(0, len(data) - args.context_length - 1, args.batch_size) # shape = (batch_size, 1)

        # Each entry is a contiguous characters(their integers) of size context_length in corpus
        batch_X = [data[idx:idx + args.context_length] for idx in batch_idxs] 
        
        # Each entry is the next character(it's integer) after context_length of contiguous character
        batch_Y = [data[idx + args.context_length] for idx in batch_idxs]
        
        
        # convert the batch to tensors
        batch_X_t = torch.tensor(
            np.array(batch_X), dtype=torch.long, device=args.device) # shape = (batch_size, context_length)
        batch_Y_t = torch.tensor(
            np.array(batch_Y), dtype=torch.long, device=args.device) # shape = (batch_size, 1)
        
        # forward pass
        # For the given context of "context_length" characters, predict the next character
        logits = model(batch_X_t) # shape = (batch_size, vocab_size)
        outs = F.log_softmax(logits, dim=1)  # softmax ensures sum of prob of each char is 1

        # compute the loss
        # Negative log likelihood loss 
        # Loss = sum(-log outs[i,batch_Y_t[i]]), i is the index

                         #pred  #target
        loss = F.nll_loss(outs, batch_Y_t) 

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print the loss
        if i % args.log_freq == 0:
            print(f'Iteration: {i}, Loss: {loss.item()}')
        
        # save the model
        if i % (10 * args.log_freq) == 0:
            torch.save(model.state_dict(), f'models/model_{args.model}_{args.context_length}.pt')
    model.eval()


def generate(model, char_to_idx, idx_to_char, args, prompt):
    model.eval()
    preds = []
    temp = args.temperature
    with torch.no_grad():
        buffer = [char_to_idx[ch] for ch in prompt][:args.context_length]

        # Generate next 1000 characters
        for i in range(1000):
            # predict the next character for given context of "buffer"
            logits = model(torch.tensor(
                np.array([buffer]), dtype=torch.long).to(args.device))
            # pred = F.softmax(logits, dim=1).argmax(dim=1).item()
            pred = torch.multinomial(F.softmax(logits / temp, dim=1), 1).item() # take the index of max_prob
            preds.append(idx_to_char[pred])

            # update the buffer by removing first character and adding pred at end
            buffer = buffer[1:] + [pred]
    preds = ''.join(preds)
    return preds
