import numpy as np

import torch
import torch.nn.functional as F


def train(model, optimizer, data, args):
    model.train()
    h,c = model.init_hidden(args.batch_size,args.device)
    for i in range(1, args.num_iterations + 1):
        # get the batch
        # gets random integers of size "batch_size" in the range [0,len(data) - args.context_length - 1] 
        batch_idxs = np.random.randint(0, len(data) - args.context_length - 1, args.batch_size) # shape = (batch_size, 1)

        # Each entry is a contiguous characters(their integers) of size context_length in corpus
        batch_X = [data[idx:idx + args.context_length] for idx in batch_idxs] 
        
        # Each entry is the next character(it's integer) 
        batch_Y = [data[idx+1: idx+1+args.context_length] for idx in batch_idxs]
        
        
        # convert the batch to tensors
        batch_X_t = torch.tensor(
            np.array(batch_X), dtype=torch.long, device=args.device) # shape = (batch_size, context_length)
        batch_Y_t = torch.tensor(
            np.array(batch_Y), dtype=torch.long, device=args.device) # shape = (batch_size, context_length)
        
        h,c = h.data,c.data

        # forward pass
        # For the given context of "context_length" characters, predict the next character after each char
        lstm_out, (h,c) = model(batch_X_t,h,c) # lstm_out.shape = (batch_size*cotext_length, vocab_size)
        outs = F.log_softmax(lstm_out, dim=1)  # softmax ensures sum of prob of each char is 1

        # compute the loss
        # Negative log likelihood loss 
        # batch_Y_t.view(-1).shape = (batch_size*context_length)
                         #pred  #target
        loss = F.nll_loss(outs, batch_Y_t.view(-1)) 

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
    h,c = model.init_hidden(1,args.device)

    with torch.no_grad():
        buffer = [char_to_idx[ch] for ch in prompt][:-1]
        X = torch.tensor(
            np.array(buffer), dtype=torch.long, device=args.device)
        X = X.unsqueeze(dim = 0) # shape = (1,len(buffer))
        _ , (h,c) = model(X,h,c)
        
        last_char_index = char_to_idx[prompt[-1]]

        # Generate next 1000 characters
        for i in range(1000):
            # predict the next character for given context of "buffer"
            x = torch.tensor([last_char_index], dtype=torch.long).reshape(1,1).to(args.device) # shape = (1,1)
            output, (h,c) = model(x,h,c) # output.shape = (bacth_size = 1,seq_len = 1,vocab_size = 61)
            # pred = F.softmax(logits, dim=1).argmax(dim=1).item()

            # output.view(-1) shape = [61]
            pred = torch.multinomial(F.softmax(output.view(-1) / temp, dim=0), 1).item() # take the index of max_prob
            preds.append(idx_to_char[pred])

            last_char_index = pred

            # update the buffer by removing first character and adding pred at end
            #buffer = buffer[1:] + [pred]
    preds = ''.join(preds)
    return preds
