import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# not vaild code

# ----------------------------------------------------------------------
# 1. Linear4bit: Simulating 4-bit Quantization for Linear Layers
# ----------------------------------------------------------------------

class Linear4bit(nn.Module):
    """
    A simplified implementation of a 4-bit quantized linear layer.
    - Weight is stored as 32-bit floats but quantized to 4-bit integers with scaling factors.
    - During forward pass, weights are dequantized back to float.
    Note: In practice, specialized libraries or CUDA kernels are required for efficient 4-bit operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Store original float weights
        self.weight_original = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_4bit", None)    # 4-bit integers (stored as uint8 for simplicity)
        self.register_buffer("scale", None)          # Scaling factor
        self.register_buffer("zero_point", None)     # Zero-point
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight_original, a=math.sqrt(5))
        self.quantize_weight()

        if self.use_bias:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weight(self):
        """
        Quantize the original float weights to 4-bit integers.
        """
        with torch.no_grad():
            w = self.weight_original.data

            # Compute min and max for scaling
            w_min = w.min()
            w_max = w.max()
            range_ = w_max - w_min
            range_ = torch.clamp(range_, min=1e-8)

            # Compute scale (quantization step size)
            self.scale = range_ / 15.0  # 4-bit: 16 levels (0-15)
            self.scale = self.scale.unsqueeze(0)

            # Compute zero point
            self.zero_point = (-w_min / self.scale).round()
            self.zero_point = self.zero_point.unsqueeze(0)

            # Quantize weights
            w_fp = w / self.scale + self.zero_point
            w_q = torch.clamp(w_fp, 0, 15).round()

            # Store as uint8 (values 0-15)
            self.weight_4bit = w_q.to(torch.uint8)

    def dequantize_weight(self):
        """
        Dequantize the 4-bit integer weights back to float.
        """
        w_q = self.weight_4bit.float()
        w_rec = (w_q - self.zero_point) * self.scale  # (out_features, in_features)
        return w_rec

    def forward(self, x):
        """
        Forward pass using dequantized weights.
        Args:
            x: Input tensor of shape (B, *, in_features)
        Returns:
            Output tensor of shape (B, *, out_features)
        """
        # Dequantize weights
        w = self.dequantize_weight()
        out = x.matmul(w.t())

        if self.use_bias and self.bias is not None:
            out = out + self.bias
        return out

# ----------------------------------------------------------------------
# 2. Rotary Embedding (Simplified for Llama-like Models)
# ----------------------------------------------------------------------

class MllamaRotaryEmbedding(nn.Module):
    """
    A simplified implementation of Rotary Positional Embedding.
    - Applies rotation transformations to query and key tensors.
    """
    def __init__(self, rotary_dim=128):
        super().__init__()
        self.rotary_dim = rotary_dim  # Number of dimensions to apply rotation

    def rotate_half(self, x):
        """
        Applies a 2D rotation to the first half of the rotary dimensions.
        """
        B, N, H = x.shape
        half = self.rotary_dim
        x1 = x[..., :half]
        x2 = x[..., half:2*half]

        # Define rotation angle (theta). In practice, this should vary with position.
        theta = 0.01  # Example angle
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        x1_rot = x1 * cos_t - x2 * sin_t
        x2_rot = x1 * sin_t + x2 * cos_t

        x_rot = torch.cat([x1_rot, x2_rot, x[..., 2*half:]], dim=-1)
        return x_rot

    def forward(self, x):
        """
        Forward pass applying rotary embeddings.
        Args:
            x: Input tensor of shape (B, N, H)
        Returns:
            Rotated tensor of shape (B, N, H)
        """
        # Apply rotation to the first rotary_dim * 2 dimensions
        if x.size(-1) < self.rotary_dim * 2:
            # If hidden size is smaller than rotary_dim * 2, return as is
            return x
        return self.rotate_half(x)

# ----------------------------------------------------------------------
# 3. Vision Model Components
# ----------------------------------------------------------------------

class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    """
    Embeds aspect ratio IDs.
    """
    def __init__(self, num_aspect_ratios=9, embedding_dim=5120):
        super().__init__()
        self.embedding = nn.Embedding(num_aspect_ratios, embedding_dim)

    def forward(self, aspect_ratio_ids):
        return self.embedding(aspect_ratio_ids)

class MllamaPrecomputedPositionEmbedding(nn.Module):
    """
    Simulates precomputed tile embeddings.
    """
    def __init__(self, num_tiles=9, embedding_dim=8197120):
        super().__init__()
        self.tile_embedding = nn.Embedding(num_tiles, embedding_dim)

    def forward(self, tile_ids):
        return self.tile_embedding(tile_ids)

class MllamaVisionSdpaAttention(nn.Module):
    """
    Self-Attention mechanism for Vision Transformer.
    """
    def __init__(self, hidden_size=1280):
        super().__init__()
        self.q_proj = Linear4bit(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear4bit(hidden_size, hidden_size, bias=False)
        self.v_proj = Linear4bit(hidden_size, hidden_size, bias=False)
        self.o_proj = Linear4bit(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        q = self.q_proj(hidden_states)  # (B, N, H)
        k = self.k_proj(hidden_states)  # (B, N, H)
        v = self.v_proj(hidden_states)  # (B, N, H)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)  # (B, N, N)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_probs = F.softmax(scores, dim=-1)  # (B, N, N)
        context = torch.matmul(attn_probs, v)  # (B, N, H)

        out = self.o_proj(context)  # (B, N, H)
        return out

class MllamaVisionMLP(nn.Module):
    """
    MLP block for Vision Transformer.
    """
    def __init__(self, hidden_size=1280, intermediate_size=5120, activation=F.gelu):
        super().__init__()
        self.fc1 = Linear4bit(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear4bit(intermediate_size, hidden_size, bias=True)
        self.activation_fn = activation

    def forward(self, x):
        x = self.fc1(x)  # (B, N, intermediate_size)
        x = self.activation_fn(x)
        x = self.fc2(x)  # (B, N, hidden_size)
        return x

class MllamaVisionEncoderLayer(nn.Module):
    """
    Single layer of the Vision Transformer Encoder.
    """
    def __init__(self, hidden_size=1280, intermediate_size=5120):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.self_attn = MllamaVisionSdpaAttention(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = MllamaVisionMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention
        normed = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(normed, attention_mask)
        hidden_states = hidden_states + attn_output

        # MLP
        normed = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed)
        hidden_states = hidden_states + mlp_output

        return hidden_states

class MllamaVisionEncoder(nn.Module):
    """
    Vision Transformer Encoder composed of multiple layers.
    """
    def __init__(self, num_layers, hidden_size=1280, intermediate_size=5120):
        super().__init__()
        self.layers = nn.ModuleList([
            MllamaVisionEncoderLayer(hidden_size, intermediate_size) for _ in range(num_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class MllamaVisionModel(nn.Module):
    """
    Vision Model that processes image inputs to extract vision features.
    """
    def __init__(self, 
                 patch_embed_in_channels=3,
                 patch_embed_out_channels=1280,
                 patch_kernel_size=14,
                 patch_stride=14,
                 patch_padding=0,
                 num_global_layers=8,
                 num_local_layers=32,
                 hidden_size=1280,
                 intermediate_size=5120,
                 num_aspect_ratios=9):
        super().__init__()

        # Patch Embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=patch_embed_in_channels,
            out_channels=patch_embed_out_channels,
            kernel_size=patch_kernel_size,
            stride=patch_stride,
            padding=patch_padding,
            bias=False
        )

        # Layer Normalization
        self.layernorm_pre = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layernorm_post = nn.LayerNorm(hidden_size, eps=1e-5)

        # Positional Embeddings
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(
            num_tiles=num_aspect_ratios,
            embedding_dim=8197120
        )
        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            num_aspect_ratios=num_aspect_ratios,
            embedding_dim=5120
        )
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            num_aspect_ratios=num_aspect_ratios,
            embedding_dim=5120
        )

        # Local Transformer
        self.transformer = MllamaVisionEncoder(
            num_layers=num_local_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )

        # Global Transformer
        self.global_transformer = MllamaVisionEncoder(
            num_layers=num_global_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )

    def forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask=None):
        """
        Forward pass for Vision Model.
        Args:
            pixel_values: Tensor of shape (B, 3, H, W)
            aspect_ratio_ids: Tensor of shape (B,)
            aspect_ratio_mask: Optional tensor for masking
        Returns:
            vision_features: Tensor of shape (B, N, H)
        """
        # Patch Embedding
        x = self.patch_embedding(pixel_values)  # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)        # (B, H'*W', C)

        x = self.layernorm_pre(x)

        # Note: Actual implementation would add positional embeddings here.
        # Aspect ratio IDs, gated positional embeddings, etc., should be integrated appropriately.
        # Simplified for this example.

        # Local Transformer
        x = self.transformer(x, attention_mask=aspect_ratio_mask)  # (B, N, H)

        x = self.layernorm_post(x)

        # Global Transformer
        x = self.global_transformer(x, attention_mask=aspect_ratio_mask)  # (B, N, H)

        return x  # (B, N, hidden_size)

# ----------------------------------------------------------------------
# 4. Language Model Components
# ----------------------------------------------------------------------

class MllamaTextRMSNorm(nn.Module):
    """
    RMSNorm as used in Llama models.
    """
    def __init__(self, dimension, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()  # (B, N, 1)
        return self.weight * x / norm_x

class MllamaTextSelfSdpaAttention(nn.Module):
    """
    Self-Attention mechanism for Text Transformer.
    """
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.q_proj = Linear4bit(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear4bit(hidden_size, hidden_size // 4, bias=False)
        self.v_proj = Linear4bit(hidden_size, hidden_size // 4, bias=False)
        self.o_proj = Linear4bit(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        q = self.q_proj(hidden_states)  # (B, N, H)
        k = self.k_proj(hidden_states)  # (B, N, H/4)
        v = self.v_proj(hidden_states)  # (B, N, H/4)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1)**0.5)  # (B, N, N)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_probs = F.softmax(scores, dim=-1)  # (B, N, N)
        context = torch.matmul(attn_probs, v)   # (B, N, H/4)

        out = self.o_proj(context)  # (B, N, H)
        return out

class MllamaTextCrossSdpaAttention(nn.Module):
    """
    Cross-Attention mechanism for Text Transformer to attend Vision Features.
    """
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.q_proj = Linear4bit(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear4bit(hidden_size, hidden_size // 4, bias=False)
        self.v_proj = Linear4bit(hidden_size, hidden_size // 4, bias=False)
        self.o_proj = Linear4bit(hidden_size, hidden_size, bias=False)
        
        self.q_norm = MllamaTextRMSNorm(128, eps=1e-5)
        self.k_norm = MllamaTextRMSNorm(128, eps=1e-5)

    def forward(self, hidden_states, encoder_hidden_states, cross_attention_mask=None):
        """
        Args:
            hidden_states: (B, N, H)
            encoder_hidden_states: (B, M, H)
            cross_attention_mask: (B, N, M)
        Returns:
            out: (B, N, H)
        """
        q = self.q_proj(hidden_states)               # (B, N, H)
        k = self.k_proj(encoder_hidden_states)       # (B, M, H/4)
        v = self.v_proj(encoder_hidden_states)       # (B, M, H/4)

        # Apply normalization to the first 128 dimensions
        q_dim = self.q_norm(q[..., :128])
        k_dim = self.k_norm(k[..., :128])
        # Concatenate the normalized dimensions with the remaining dimensions
        q = torch.cat([q_dim, q[..., 128:]], dim=-1)
        k = torch.cat([k_dim, k[..., 128:]], dim=-1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1)**0.5)  # (B, N, M)
        if cross_attention_mask is not None:
            scores = scores + cross_attention_mask
        attn_probs = F.softmax(scores, dim=-1)  # (B, N, M)
        context = torch.matmul(attn_probs, v)   # (B, N, H/4)

        out = self.o_proj(context)              # (B, N, H)
        return out

class MllamaTextMLP(nn.Module):
    """
    MLP block for Text Transformer.
    """
    def __init__(self, hidden_size=4096, intermediate_size=14336, act_fn=F.silu):
        super().__init__()
        self.gate_proj = Linear4bit(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear4bit(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear4bit(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x):
        gated = self.gate_proj(x)  # (B, N, intermediate_size)
        up = self.up_proj(x)       # (B, N, intermediate_size)
        x = gated * self.act_fn(up)
        x = self.down_proj(x)
        return x

class MllamaSelfAttentionDecoderLayer(nn.Module):
    """
    Decoder layer with Self-Attention and MLP for Text Transformer.
    """
    def __init__(self, hidden_size=4096, intermediate_size=14336):
        super().__init__()
        self.input_layernorm = MllamaTextRMSNorm(hidden_size, eps=1e-5)
        self.self_attn = MllamaTextSelfSdpaAttention(hidden_size)
        self.post_attention_layernorm = MllamaTextRMSNorm(hidden_size, eps=1e-5)
        self.mlp = MllamaTextMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, attention_mask)
        hidden_states = hidden_states + attn_out

        # MLP
        normed2 = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed2)
        hidden_states = hidden_states + mlp_out

        return hidden_states

class MllamaCrossAttentionDecoderLayer(nn.Module):
    """
    Decoder layer with Cross-Attention and MLP for Text Transformer.
    """
    def __init__(self, hidden_size=4096, intermediate_size=14336):
        super().__init__()
        self.input_layernorm = MllamaTextRMSNorm(hidden_size, eps=1e-5)
        self.cross_attn = MllamaTextCrossSdpaAttention(hidden_size)
        self.post_attention_layernorm = MllamaTextRMSNorm(hidden_size, eps=1e-5)
        self.mlp = MllamaTextMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, encoder_hidden_states, cross_attention_mask=None):
        # Cross-Attention
        normed = self.input_layernorm(hidden_states)
        cross_attn_out = self.cross_attn(normed, encoder_hidden_states, cross_attention_mask)
        hidden_states = hidden_states + cross_attn_out

        # MLP
        normed2 = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed2)
        hidden_states = hidden_states + mlp_out

        return hidden_states

class MllamaTextModel(nn.Module):
    """
    Text Model composed of token embeddings and multiple decoder layers.
    """
    def __init__(self,
                 vocab_size=128264,
                 hidden_size=4096,
                 num_layers=40,
                 intermediate_size=14336,
                 rotary_dim=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=128004)
        self.layers = nn.ModuleList()

        # Alternate between Self-Attention and Cross-Attention layers
        for idx in range(num_layers):
            if (idx + 1) % 5 == 0:
                # Every 5th layer is a Cross-Attention layer
                self.layers.append(MllamaCrossAttentionDecoderLayer(hidden_size, intermediate_size))
            else:
                # Other layers are Self-Attention layers
                self.layers.append(MllamaSelfAttentionDecoderLayer(hidden_size, intermediate_size))

        self.norm = MllamaTextRMSNorm(hidden_size, eps=1e-5)
        self.rotary_emb = MllamaRotaryEmbedding(rotary_dim=rotary_dim)

    def forward(self,
                input_ids,
                attention_mask=None,
                encoder_hidden_states=None,
                cross_attention_mask=None):
        """
        Forward pass for Text Model.
        Args:
            input_ids: Tensor of shape (B, S)
            attention_mask: Tensor of shape (B, S)
            encoder_hidden_states: Tensor of shape (B, N, H) from Vision Model
            cross_attention_mask: Tensor of shape (B, S, N)
        Returns:
            hidden_states: Tensor of shape (B, S, H)
        """
        # Token Embedding
        x = self.embed_tokens(input_ids)  # (B, S, H)

        # Apply Rotary Embedding
        x = self.rotary_emb(x)           # (B, S, H)

        # Pass through each layer
        for layer in self.layers:
            if isinstance(layer, MllamaSelfAttentionDecoderLayer):
                x = layer(x, attention_mask)
            elif isinstance(layer, MllamaCrossAttentionDecoderLayer):
                x = layer(x, encoder_hidden_states, cross_attention_mask)
            else:
                raise ValueError("Unknown layer type.")

        # Final normalization
        x = self.norm(x)

        return x

class MllamaForCausalLM(nn.Module):
    """
    Causal Language Model for Text Generation.
    """
    def __init__(self,
                 vocab_size=128256,
                 hidden_size=4096,
                 num_layers=40,
                 intermediate_size=14336,
                 rotary_dim=128):
        super().__init__()
        self.model = MllamaTextModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            rotary_dim=rotary_dim
        )
        self.lm_head = Linear4bit(hidden_size, vocab_size, bias=False)

    def forward(self,
                input_ids,
                attention_mask=None,
                encoder_hidden_states=None,
                cross_attention_mask=None):
        """
        Forward pass for Causal LM.
        Args:
            input_ids: Tensor of shape (B, S)
            attention_mask: Tensor of shape (B, S)
            encoder_hidden_states: Tensor of shape (B, N, H) from Vision Model
            cross_attention_mask: Tensor of shape (B, S, N)
        Returns:
            logits: Tensor of shape (B, S, vocab_size)
        """
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=cross_attention_mask
        )  # (B, S, H)

        logits = self.lm_head(hidden_states)  # (B, S, vocab_size)
        return logits

# ----------------------------------------------------------------------
# 5. Multi-Modal Projector
# ----------------------------------------------------------------------

class MllamaMultiModalProjector(nn.Module):
    """
    Projects vision features to match language model dimensions.
    """
    def __init__(self, in_features=7680, out_features=4096, bias=True):
        super().__init__()
        self.projector = Linear4bit(in_features, out_features, bias=bias)

    def forward(self, vision_features):
        """
        Args:
            vision_features: Tensor of shape (B, N, H_v)
        Returns:
            projected_features: Tensor of shape (B, N, H_l)
        """
        return self.projector(vision_features)

# ----------------------------------------------------------------------
# 6. Combined Multi-Modal Model
# ----------------------------------------------------------------------

class MllamaForConditionalGeneration(nn.Module):
    """
    Multi-Modal Model combining Vision and Language components.
    """
    def __init__(self):
        super().__init__()
        # Vision Model
        self.vision_model = MllamaVisionModel()

        # Language Model
        self.language_model = MllamaForCausalLM()

        # Multi-Modal Projector
        self.multi_modal_projector = MllamaMultiModalProjector()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                pixel_values=None,
                aspect_ratio_ids=None,
                aspect_ratio_mask=None,
                cross_attention_mask=None):
        """
        Forward pass for the multi-modal model.
        Args:
            input_ids: Tensor of shape (B, S)
            attention_mask: Tensor of shape (B, S)
            pixel_values: Tensor of shape (B, 3, H, W)
            aspect_ratio_ids: Tensor of shape (B,)
            aspect_ratio_mask: Tensor of shape (B, N)
            cross_attention_mask: Tensor of shape (B, S, N)
        Returns:
            logits: Tensor of shape (B, S, vocab_size)
        """
        vision_features = None
        if pixel_values is not None:
            # Extract vision features
            vision_features = self.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask)  # (B, N, H_v)
            # Project vision features
            vision_features = self.multi_modal_projector(vision_features)  # (B, N, H_l)

        # Generate logits from language model with cross-attention to vision features
        logits = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=vision_features,
            cross_attention_mask=cross_attention_mask
        )  # (B, S, vocab_size)

        return logits

# ----------------------------------------------------------------------
# 7. Example Usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MllamaForConditionalGeneration().to(device)

    batch_size = 2
    seq_len = 16
    image_height = 224
    image_width = 224

    # Text Inputs
    input_ids = torch.randint(0, 128264, (batch_size, seq_len)).to(device)  # (B, S)
    attention_mask = torch.ones(batch_size, seq_len).to(device)             # (B, S)

    # Image Inputs
    pixel_values = torch.randn(batch_size, 3, image_height, image_width).to(device)  # (B, 3, H, W)
    aspect_ratio_ids = torch.randint(0, 9, (batch_size,)).to(device)               # (B,)
    aspect_ratio_mask = torch.ones(batch_size, (image_height // 14)*(image_width // 14)).to(device)  # Example mask
    cross_attention_mask = torch.zeros(batch_size, seq_len, (image_height // 14)*(image_width // 14)).to(device)  # (B, S, N)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask
        )

    print("Logits shape:", outputs.shape)  # Expected: (B, S, vocab_size)
