# Example config definitions
config1 = {
    "image_size": 32, 
    "patch_size": 4,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 8,
    "intermediate_size": 1536,
    "num_hidden_layers": 8,
    "qkv_bias": True,
    "attention_probs_dropout_prob": 0.1,
    "hidden_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "num_classes": 10, 
    "use_faster_attention": True,
    "positional_embedding": "1d_learned",
    "epochs" : 50
}
