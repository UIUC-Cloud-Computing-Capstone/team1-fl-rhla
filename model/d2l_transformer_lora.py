import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for linear transformations."""
    def __init__(self, base_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        """
        Args:
            base_layer: The base linear layer to adapt
            r: Rank of the low-rank decomposition
            lora_alpha: Scaling factor for LoRA weights
            lora_dropout: Dropout rate for LoRA path
        """
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Low-rank matrices - will be initialized lazily
        self.lora_A = None
        self.lora_B = None
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.initialized = False
    
    def _initialize_lora_weights(self, in_features, out_features):
        """Initialize LoRA weights lazily based on input dimensions."""
        if not self.initialized:
            # A: (in_features, r) - initialized with Kaiming uniform
            self.lora_A = nn.Parameter(torch.zeros(in_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            
            # B: (r, out_features) - initialized with zeros
            self.lora_B = nn.Parameter(torch.zeros(self.r, out_features))
            
            self.initialized = True
    
    def forward(self, x):
        # Forward through base layer
        base_output = self.base_layer(x)
        
        # Initialize LoRA weights if needed
        if not self.initialized:
            # Get dimensions from base layer
            if hasattr(self.base_layer, 'in_features'):
                in_features = self.base_layer.in_features
                out_features = self.base_layer.out_features
            else:
                # For LazyLinear, infer from input
                in_features = x.shape[-1]
                out_features = base_output.shape[-1]
            self._initialize_lora_weights(in_features, out_features)
        
        # LoRA path: x @ A @ B with scaling
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        
        return base_output + self.scaling * lora_output



class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape) # weight shape: H
        #print('self.ln.weight.shape:', self.ln.weight.shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
        


#@save
#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

#@save
#@save
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):  #@save
    """Multi-head attention with LoRA on query and value projections."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, 
                 use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.use_lora = use_lora
        
        # Create base layers
        base_W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        base_W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)
        
        # Wrap query and value with LoRA if enabled
        if use_lora:
            self.W_q = LoRALayer(base_W_q, r=lora_r, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout)
            self.W_v = LoRALayer(base_W_v, r=lora_r, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout)
        else:
            self.W_q = base_W_q
            self.W_v = base_W_v

    
    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    
    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False, use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias, use_lora=use_lora,
                                                lora_r=lora_r, lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# command to run in terminal: python model/d2l_transformer.py
B = 2 # batch size
S = 3 # sequence length
H = 4 # hidden dimension
F = 8 # feedforward dimension
NH = 4 # number of heads
dropout = 0.5   
# X = torch.ones((B, S, H))
# valid_lens = torch.tensor([2, 3])
# transformer_encoder_block = TransformerEncoderBlock(H, F, NH, dropout)
# print(transformer_encoder_block(X, valid_lens))

class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False, use_lora=True,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias,
                use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

src_vocab_size = 10
num_blks = 2
# https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
X = torch.ones((B, S), dtype=torch.long)
# 第 1 条序列：只有 前 2 个 token 是有效的
#第 2 条序列：3 个 token 全部有效
valid_lens = torch.tensor([2, 3])
transformer_encoder = TransformerEncoder(src_vocab_size, H, F, NH, num_blks, dropout)
print(transformer_encoder(X, valid_lens))

class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i,
                 use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout, use_lora=use_lora,
                                                 lora_r=lora_r, lora_alpha=lora_alpha,
                                                 lora_dropout=lora_dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout, use_lora=use_lora,
                                                 lora_r=lora_r, lora_alpha=lora_alpha,
                                                 lora_dropout=lora_dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i,
                use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

# training
# Create a wrapper class for the translation dataset
class MTFraEng:
    def __init__(self, batch_size=32, num_steps=10):
        self.data_iter, self.src_vocab, self.tgt_vocab = d2l.load_data_nmt(
            batch_size, num_steps)
        self.batch_size = batch_size

data = MTFraEng(batch_size=32)
num_hiddens, num_blks, dropout = 384, 12, 0.2
ffn_num_hiddens, num_heads = 64, 4
encoder = TransformerEncoder(
    len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
    num_blks, dropout)
decoder = TransformerDecoder(
    len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
    num_blks, dropout)
# Use EncoderDecoder instead of Seq2Seq
model = d2l.EncoderDecoder(encoder, decoder)
# Set the target padding token
model.tgt_pad = data.tgt_vocab['<pad>']

# Simple training loop (replacing Trainer)
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "_weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                              device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat.reshape(-1, len(tgt_vocab)), Y.reshape(-1))
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

# device = d2l.try_gpu()
# train_seq2seq(model, data.data_iter, lr=0.001, num_epochs=1, 
#               tgt_vocab=data.tgt_vocab, device=device)

# Example: Using LoRA with Transformer
if __name__ == "__main__":
    print("\n=== LoRA Transformer Example ===")
    
    # Create encoder with LoRA enabled (default)
    print("\n1. Creating Transformer Encoder with LoRA...")
    encoder_lora = TransformerEncoder(
        src_vocab_size, H, F, NH, num_blks, dropout, 
        use_lora=True, lora_r=4, lora_alpha=8, lora_dropout=0.1)
    
    # Initialize with dummy data
    _ = encoder_lora(X, valid_lens)
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder_lora.parameters())
    trainable_params = sum(p.numel() for p in encoder_lora.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {frozen_params:,}")
    print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Show LoRA parameters
    print("\n2. LoRA parameters in encoder:")
    for name, param in encoder_lora.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            print(f"   {name}: {param.shape}, requires_grad={param.requires_grad}")
    
    print("\n3. Base layer parameters (frozen):")
    count = 0
    for name, param in encoder_lora.named_parameters():
        if 'base_layer' in name and not param.requires_grad:
            print(f"   {name}: {param.shape}, requires_grad={param.requires_grad}")
            count += 1
            if count >= 3:  # Just show first 3
                print("   ...")
                break
    
    print("\n=== LoRA successfully added to query and value modules! ===")