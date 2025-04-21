import math
import torch
from torch import nn


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    """
    Base class for different positional embedding strategies.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.initializer_range = config.get("initializer_range", 0.02)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
                Note: seq_len includes the CLS token
        Returns:
            Positional embeddings to be added to the input
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def initialize_weights(self):
        """
        Initialize position embeddings for each specific implementation
        """
        pass


class NoPositionalEmbedding(PositionalEmbedding):
    """
    No positional embedding - returns zeros.
    """
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return torch.zeros_like(x)


class Learned1DPositionalEmbedding(PositionalEmbedding):
    """
    1D learned positional embeddings (default ViT approach).
    """
    def __init__(self, config):
        super().__init__(config)
        # Position embeddings for CLS token + patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.hidden_size)
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize the position embeddings
        self.position_embeddings.data = nn.init.trunc_normal_(
            self.position_embeddings.data.to(torch.float32),
            mean=0.0,
            std=self.initializer_range,
        ).to(self.position_embeddings.dtype)
    
    def forward(self, x):
        return self.position_embeddings


class Learned2DPositionalEmbedding(PositionalEmbedding):
    """
    2D learned positional embeddings (as in ViT Appendix D.4).
    """
    def __init__(self, config):
        super().__init__(config)
        grid_size = int(math.sqrt(self.num_patches))
        
        # Separate embeddings for height and width dimensions
        self.pos_embed_h = nn.Parameter(torch.randn(1, grid_size, self.hidden_size // 2))
        self.pos_embed_w = nn.Parameter(torch.randn(1, grid_size, self.hidden_size // 2))
        
        # Special embedding for CLS token
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize each parameter separately
        self.pos_embed_h.data = nn.init.trunc_normal_(
            self.pos_embed_h.data.to(torch.float32),
            mean=0.0,
            std=self.initializer_range,
        ).to(self.pos_embed_h.dtype)
        
        self.pos_embed_w.data = nn.init.trunc_normal_(
            self.pos_embed_w.data.to(torch.float32),
            mean=0.0,
            std=self.initializer_range,
        ).to(self.pos_embed_w.dtype)
        
        self.cls_pos_embedding.data = nn.init.trunc_normal_(
            self.cls_pos_embedding.data.to(torch.float32),
            mean=0.0,
            std=self.initializer_range,
        ).to(self.cls_pos_embedding.dtype)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        grid_size = int(math.sqrt(self.num_patches))
        
        # Handle CLS token separately
        cls_pos_embed = self.cls_pos_embedding
        
        # Generate 2D positional embeddings for patch tokens
        h_pos = self.pos_embed_h.unsqueeze(2).expand(-1, -1, grid_size, -1)
        w_pos = self.pos_embed_w.unsqueeze(1).expand(-1, grid_size, -1, -1)
        
        # Combine height and width embeddings
        hw_pos = torch.cat([h_pos, w_pos], dim=-1)
        # Reshape to (1, num_patches, hidden_size)
        hw_pos = hw_pos.reshape(1, self.num_patches, self.hidden_size)
        
        # Combine CLS positional embedding with patch positional embeddings
        pos_embed = torch.cat([cls_pos_embed, hw_pos], dim=1)
        
        return pos_embed


class SinusoidalPositionalEmbedding(PositionalEmbedding):
    """
    Sinusoidal positional embedding from 'Attention Is All You Need'.
    """
    def __init__(self, config):
        super().__init__(config)
        # Pre-compute the sinusoidal embeddings
        position = torch.arange(self.num_patches + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2) * 
                             -(math.log(10000.0) / self.hidden_size))
        
        pos_embed = torch.zeros(1, self.num_patches + 1, self.hidden_size)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pos_embed', pos_embed)
    
    def forward(self, x):
        return self.pos_embed


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        
        # Number of patches
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2
        
        # Select positional embedding type based on config
        pos_embed_type = config.get("positional_embedding", "1d_learned")
        
        if pos_embed_type == "none":
            self.position_embedding = NoPositionalEmbedding(config)
        elif pos_embed_type == "1d_learned":
            self.position_embedding = Learned1DPositionalEmbedding(config)
        elif pos_embed_type == "2d_learned":
            self.position_embedding = Learned2DPositionalEmbedding(config)
        elif pos_embed_type == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEmbedding(config)
        else:
            raise ValueError(f"Unknown positional embedding type: {pos_embed_type}")
            
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.initialize_cls_token()

    def initialize_cls_token(self):
        # Initialize the CLS token
        self.cls_token.data = nn.init.trunc_normal_(
            self.cls_token.data.to(torch.float32),
            mean=0.0,
            std=self.config.get("initializer_range", 0.02),
        ).to(self.cls_token.dtype)

    def forward(self, x):
        x = self.patch_embeddings(x)
        
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        pos_embed = self.position_embedding(x)
        x = x + pos_embed
        
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module with two hidden layers.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation_1 = NewGELUActivation()
        self.dense_3 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation_1(x)
        x = self.dense_3(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        logits = self.classifier(encoder_output[:, 0, :])

        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # Note: We removed the Embeddings class initialization here since
        # each position embedding class handles its own initialization