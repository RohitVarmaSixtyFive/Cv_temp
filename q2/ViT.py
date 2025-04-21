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


class DifferentialMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        # Calculate sequence length including CLS token
        self.seq_len = (config['image_size'] // config['patch_size']) ** 2 + 1
        self.head_dim = self.hidden_size // self.num_heads
        
        # Create separate projections for differential attention
        self.W_q = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.W_k = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config.get("attention_probs_dropout_prob", 0.1))
        
        # Initialize lambda as a learnable parameter
        self._lambda = nn.Parameter(torch.tensor(0.5))
        
        # Optional: Group norm for stabilizing attention outputs
        self.use_group_norm = config.get("use_group_norm", True)
        if self.use_group_norm:
            self.group_norm = nn.GroupNorm(num_groups=self.num_heads, num_channels=self.num_heads * self.head_dim)

    def forward(self, x, output_attentions=False):
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to queries, keys, and values
        q_combined = self.W_q(x)  # (batch_size, seq_len, 2*hidden_size)
        k_combined = self.W_k(x)  # (batch_size, seq_len, 2*hidden_size)
        v = self.W_v(x)           # (batch_size, seq_len, hidden_size)
        
        # Split into two sets of queries and keys
        q1, q2 = torch.split(q_combined, self.hidden_size, dim=-1)
        k1, k2 = torch.split(k_combined, self.hidden_size, dim=-1)
        
        # Reshape for multi-head attention
        q1 = q1.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        q2 = q2.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = k1.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_scores2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply softmax separately to both attention mechanisms
        attn_probs1 = nn.functional.softmax(attn_scores1, dim=-1)
        attn_probs2 = nn.functional.softmax(attn_scores2, dim=-1)
        
        # Differential attention calculation
        diff_attn_probs = attn_probs1 - self._lambda * attn_probs2
        diff_attn_probs = self.dropout(diff_attn_probs)
        
        # Apply attention to values
        context = torch.matmul(diff_attn_probs, v)  # (batch, heads, seq_len, head_dim)
        
        # Apply group normalization if enabled
        if self.use_group_norm:
            
            context_reshaped = context.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, self.num_heads)
            
            # Reshape context to (B, C, L) where C = num_heads * head_dim
            context_reshaped = context_reshaped.view(batch_size, self.num_heads * self.head_dim, seq_len)
                        
            # Apply GroupNorm
            context_normed = self.group_norm(context_reshaped)

            # Reshape back to (B, L, C)
            context = context_normed.permute(0, 2, 1).contiguous()
        
        # Reshape output back to (batch, seq_len, hidden_size)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.W_o(context)
        
        if output_attentions:
            return (output, diff_attn_probs)
        else:
            return (output, None)


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
    A single transformer block with support for standard or differential attention.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        self.use_differential_attention = config.get("use_differential_attention", False)
        
        if self.use_differential_attention:
            self.attention = DifferentialMultiHeadAttention(config)
        else:
            print(f"There is an error brother, re-enter hyperparams")
            
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        
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


class DiffViT(nn.Module):
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