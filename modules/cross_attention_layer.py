import torch
import torch.nn as nn

class CrossAttentionEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CrossAttentionEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.d_model = d_model
        self.nhead = nhead
        # Replace self-attention with cross-attention
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    
    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass of the cross-attention layer.
        
        Args:
            src: Input embedding for queries (e.g., RGB).
            memory: Input embedding for keys and values (e.g., Depth).
            src_mask: Optional attention mask for queries.
            src_key_padding_mask: Optional padding mask for queries.
        
        Returns:
            Output after cross-attention and feed-forward sublayers.
        """

        # Norm + Cross-Attention (Q: src, K/V: memory)
        src2 = self.cross_attention(
            query=src, key=memory, value=memory, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-Forward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class CrossAttentionTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CrossAttentionTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src, memory, mask=None, key_padding_mask=None):
        """
        Custom forward to handle src and memory as separate inputs.
        """
        output = src
        for layer in self.layers:
            output = layer(output, memory, src_mask=mask, src_key_padding_mask=key_padding_mask)
        return self.norm(output)