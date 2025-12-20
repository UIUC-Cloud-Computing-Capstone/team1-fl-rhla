import torch
from torch import nn

class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()

        # LazyLinear will change size
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        # shape of X is (batch_size, seq_len, num_hiddens)
        print('X.shape:', X.shape)
        print('self.dense1(X).shape:', self.dense1(X).shape)
        print('self.relu(self.dense1(X)).shape:', self.relu(self.dense1(X)).shape)


        print('self.dense2(self.relu(self.dense1(X))).shape:', self.dense2(self.relu(self.dense1(X))).shape)
        return self.dense2(self.relu(self.dense1(X)))

ffn_num_hiddens = 4
ffn_num_outputs = 8
pwffn = PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs)
X = torch.ones((2, 3, ffn_num_hiddens))
print(pwffn(X).shape)

# command to run in terminal: python tests/test_shape.py

