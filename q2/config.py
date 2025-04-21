config1 = {
    "image_size": 32,
    "patch_size": 4,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_hidden_layers": 8,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 1000,
    "use_differential_attention": True,  
    "positional_embedding": "1d_learned"
}

config2 = {
    "image_size": 32,
    "patch_size": 4,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 1000,
    "use_differential_attention": True,  
    "positional_embedding": "1d_learned"
}

config3 = {
    "image_size": 32,
    "patch_size": 2,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 1000,
    "use_differential_attention": True,  
    "positional_embedding": "1d_learned"
}

config4 = {
    "image_size": 32,
    "patch_size": 8,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_hidden_layers": 8,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 1000,
    "use_differential_attention": True,  
    "positional_embedding": "1d_learned"
}

config5 = {
    "image_size": 32,
    "patch_size": 4,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_hidden_layers": 8,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 1000,
    "use_differential_attention": True,  
    "positional_embedding": "1d_learned"
}

config_2d = {
    "image_size": 32,
    "patch_size": 4,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_hidden_layers": 8,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 1000,
    "use_differential_attention": True,  
    "positional_embedding": "2d_learned"
}

config_sinosoidal = {
    "image_size": 32,
    "patch_size": 2,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_hidden_layers": 8,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 1000,
    "use_differential_attention": True,  
    "positional_embedding": "sinusoidal"
}

config_none = {
    "image_size": 32,
    "patch_size": 2,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_hidden_layers": 8,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 1000,
    "use_differential_attention": True,  
    "positional_embedding": "none"
}