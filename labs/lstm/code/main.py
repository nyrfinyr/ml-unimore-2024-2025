import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


#########################################################################
# TO-DO: Implement the __getitem__ of the Shakespeare class.
# Important points: __getitem__ is called at each training iteration
# So, we need to return the data and ground-truth label. The data is in
# the form of one-hot vectors and ground-truth is the index of next char
# Our RNN operates on an input sequence of a specified length (seq_length)
# so we need to return a sequence of one-hot vector and the indices of
# their corresponding next character
#########################################################################

class Shakespeare(Dataset):

    def __init__(self, text_data, seq_length, data_size, chars, vocab_size, char_to_ix, ix_to_char):
        super().__init__()

        self.seq_length = seq_length
        self.data = text_data
        self.data_size = data_size
        self.chars = chars
        self.vocab_size = vocab_size
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char

        # hint: preprocess data for faster loading

    def __len__(self):
        return self.data_size - self.seq_length


    def __getitem__(self, index):
        start_idx = index
        end_idx = start_idx + self.seq_length

        input_seq = self.data[start_idx:end_idx]  # X (sequence of characters)
        target_char = self.data[start_idx + self.seq_length]  # y (next character)

        target_index = self.char_to_ix[target_char]

        # Convert to PyTorch tensors
        input_tensor = torch.tensor((self.seq_length, self.vocab_size, self.vocab_size))
        for i in range(self.seq_length):
            input_tensor[i] = int_to_onehot()

        target_tensor = torch.tensor(target_index, dtype=torch.long)  # Single value: index of next character

        return input_tensor, target_tensor


def int_to_onehot(code, max_code):
    return torch.eye(max_code)[:,code]


def main():
    # read the contents of the text file
    data = open('./tinyshakespeare.txt', 'r').read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    # Create a dictionary mapping each character to a unique id and vice versa
    char_to_ix = {e : i for i, e in enumerate(chars)}
    ix_to_char = {i : e for i, e in enumerate(chars)}




if __name__ == '__main__':
    main()