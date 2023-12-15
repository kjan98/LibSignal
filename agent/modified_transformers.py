import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    # def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
    def __init__(self, h, d_k, d_v, d_model, d_ffnn, rate) # TODO: d_k and d_v aren't used
        super(TransformerBlock, self).__init__()
        
        """
        Args:
          h: number of attention heads
          d_k: dimension of k input
          d_v: dimension of value input
          d_model: dimension of model output
          d_ffnn: dimesnion of inner ffnn
          rate: dropout rate
        """
        self.attention = nn.MultiHeadAttention(d_model,
                                               num_heads=h,
                                               dropout=rate,
                                               kdim=d_k,
                                               vdim=d_v,
                                               batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(d_model, d_ffnn),
                          nn.ReLU(),
                          nn.Linear(d_ffnn, d_model)
        )

        self.dropout1 = nn.Dropout(rate) # 0.2
        self.dropout2 = nn.Dropout(rate)

    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """
        
        attention_out = self.attention(key,query,value) 
        attention_residual_out = attention_out + value 
        norm1_out = self.norm1(attention_residual_out)

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        drop2_out = self.dropout2(feed_fwd_residual_out)
        norm2_out = self.norm2(drop2_out)

        return norm2_out



class TransformerEncoder(nn.Module):
    """
    Args:
        h: number of attention heads
        d_k: dimension of k input
        d_v: dimension of value input
        d_model: dimension of model output
        d_ffnn: dimesnion of inner ffnn
        rate: dropout rate
        n: number of layers
    Returns:
        out: output of the encoder
    """
    def pos

    # def __init__(self, seq_len, vocab_size, embed_dim, num_layers=6, expansion_factor=4, n_heads=8):
    def __init__(self, h, d_k, d_v, d_model, d_ffnn, rate, n):
        super(TransformerEncoder, self).__init__()
        
        # self.embedding_layer = Embedding(vocab_size, embed_dim)
        # self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(h, d_k, d_v, d_model, d_ffnn, rate) for i in range(num_layers)])
    
    def forward(self, x):
        # embed_out = self.embedding_layer(x)
        # out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(x,x,x)

        return out  #32x10x512


    #         self.k = 20 #change to be dic_feature_dim[D_TIMESTEP] config call
    # self.h = 4 # num attn heads
    # self.d_k = 8 # dim of linearly projected keys and heads
    # self.d_v = 8 # dim of linearly projected values
    # self.d_ff = 2048  # Dimensionality of the inner fully connected layer
    # self.d_model = 20  # Dimensionality of the model sub-layers' outputs; pick 20 bc that matches embedding dim set by Conv2d of auths
    # self.n = 6  # Number of layers in the encoder stack