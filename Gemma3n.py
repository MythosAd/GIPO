# MatFormer: https://arxiv.org/pdf/2310.07707	
# AltUp Layers: https://arxiv.org/pdf/2301.13310	
# Laurel Blocks: https://arxiv.org/pdf/2411.07501	

import torch	
import torch.nn as nn	
import torch.nn.functional as F	
import math	
from typing import Optional, Tuple	

class LayerNorm(nn.Module):	
    """RMS Layer normalization matching the STABLEHLO_COMPOSITE operations"""	
    def __init__(self, dim, eps=1e-6):	
        super().__init__()	
        self.weight = nn.Parameter(torch.ones(dim))	
        self.eps = eps	

    def forward(self, x):	
        # Compute RMS normalization: x / sqrt(mean(x^2) + eps) * weight
        x_squared = x * x	
        mean_squared = x_squared.mean(dim=-1, keepdim=True)	
        rms = torch.sqrt(mean_squared + self.eps)  # Fixed: added eps before sqrt
        return (x / rms) * self.weight	

class RotaryPositionEmbedding(nn.Module):	
    """RoPE implementation based on the sin/cos operations in the model"""	
    def __init__(self, dim, max_seq_len=4096):	
        super().__init__()	
        self.dim = dim	
        assert dim % 2 == 0, "Dimension must be even for RoPE"

        # Precompute sin/cos embeddings	
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))	
        position = torch.arange(max_seq_len).float()	
        sincos = torch.einsum('i,j->ij', position, inv_freq)	
        self.register_buffer('sin', sincos.sin())	
        self.register_buffer('cos', sincos.cos())	

    def forward(self, x, position_ids):	
        # x shape: [batch, seq_len, num_heads, head_dim]	
        batch, seq_len, num_heads, head_dim = x.shape	

        # Split into two halves for rotation	
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]	

        # Handle position_ids - ensure it's 1D
        if position_ids.dim() > 1:
            position_ids = position_ids.squeeze()
        if position_ids.dim() == 0:
            position_ids = position_ids.unsqueeze(0)
            
        # Get sin/cos for the positions
        cos = self.cos[position_ids]  # [seq_len, head_dim//2]
        sin = self.sin[position_ids]  # [seq_len, head_dim//2]
        
        # Expand to match x1, x2 dimensions: [batch, seq_len, num_heads, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        cos = cos.expand(batch, -1, num_heads, -1)
        sin = sin.expand(batch, -1, num_heads, -1)

        # Apply rotation	
        rx1 = x1 * cos - x2 * sin	
        rx2 = x2 * cos + x1 * sin	

        return torch.cat([rx1, rx2], dim=-1)	

class Attention(nn.Module):	
    """Multi-head attention with RoPE"""	
    def __init__(self, dim, num_heads=8, head_dim=256):	
        super().__init__()	
        self.num_heads = num_heads	
        self.head_dim = head_dim	
        self.scale = 1.0 / math.sqrt(head_dim)  # Standard attention scaling

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)	
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False) 	
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)	
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)	

        # Query/Key normalization for stability
        self.q_norm = LayerNorm(head_dim)	
        self.k_norm = LayerNorm(head_dim)	

        # RoPE	
        self.rope = RotaryPositionEmbedding(head_dim)	

    def forward(self, x, position_ids, kv_cache=None):	
        batch, seq_len, _ = x.shape	

        # Project to Q, K, V	
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)	
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)	
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)	

        # Apply layer norms to Q and K for stability
        q = self.q_norm(q)	
        k = self.k_norm(k)	

        # Apply RoPE positional encoding
        q = self.rope(q, position_ids)	
        k = self.rope(k, position_ids)	

        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask for autoregressive generation
        if kv_cache is None:  # Only mask during training/prefill
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            scores = scores + causal_mask.to(scores.device)

        # Apply softmax and compute attention output
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to [batch, seq_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        
        return self.o_proj(attn_output)

class MLP(nn.Module):	
    """SwiGLU Feed-forward network"""	
    def __init__(self, dim, hidden_dim=8192):	
        super().__init__()	
        # SwiGLU uses two projections: gate and up
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)	
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)	
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)	

    def forward(self, x):	
        # SwiGLU: gate(x) * up(x) where gate uses SiLU activation
        gate = F.silu(self.gate_proj(x))  # Changed from GELU to SiLU for SwiGLU
        up = self.up_proj(x)	
        return self.down_proj(gate * up)

class LaurelBlock(nn.Module):	
    """Laurel residual connection block - low-rank bottleneck"""	
    def __init__(self, dim, bottleneck_dim=64):	
        super().__init__()	
        # Low-rank projection down and back up
        self.down_proj = nn.Linear(dim, bottleneck_dim, bias=False)	
        self.up_proj = nn.Linear(bottleneck_dim, dim, bias=False)	
        self.norm = LayerNorm(dim)	

    def forward(self, x):	
        # Store residual connection
        residual = x	
        # Low-rank transformation
        x = self.down_proj(x)	
        x = self.up_proj(x)	
        x = self.norm(x)	
        # Add residual connection
        return residual + x	

class AltupRouter(nn.Module):	
    """ALTUP routing mechanism for expert selection"""	
    def __init__(self, dim, num_experts=4):	
        super().__init__()	
        self.norm = LayerNorm(dim)	
        self.router = nn.Linear(dim, num_experts, bias=False)	
        self.scale = 3.0  # Temperature scaling for sharper routing

    def forward(self, x):	
        # Normalize input before routing
        x = self.norm(x)	
        logits = self.router(x)	
        # Apply temperature scaling and tanh activation
        logits = logits * self.scale	
        return torch.tanh(logits)
    
class TransformerLayer(nn.Module):
    """Single transformer layer with ALTUP routing and correction"""
    def __init__(self, dim, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Layer normalization before each sub-layer
        self.pre_attn_norm = LayerNorm(dim)
        self.attention = Attention(dim)
        self.post_attn_norm = LayerNorm(dim)
        
        # Laurel block for additional residual connections
        self.laurel = LaurelBlock(dim)
        
        # MLP with pre/post normalization
        self.pre_mlp_norm = LayerNorm(dim)
        self.mlp = MLP(dim)
        self.post_mlp_norm = LayerNorm(dim)
        
        # ALTUP routing mechanisms
        self.altup_router_predict = AltupRouter(dim)
        self.altup_router_correct = AltupRouter(dim)
        
        # ALTUP expert projections (3 experts)
        self.altup_proj = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(3)
        ])
        self.altup_unproj = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(3)
        ])
        
        # Per-layer embedding integration
        self.per_layer_gate = nn.Linear(dim, 256, bias=False)
        self.per_layer_proj = nn.Linear(256, dim, bias=False)
        self.per_layer_norm = LayerNorm(dim)
        
        # Scaling factors for residual connections
        self.residual_scale = math.sqrt(1.0/3.0)  # Cleaner constant definition
        
    def altup_predict(self, x):
        """ALTUP prediction step: route input through multiple experts"""
        # Get routing probabilities from input
        router_weights = self.altup_router_predict(x)
        router_probs = F.softmax(router_weights, dim=-1)
        
        # Apply expert transformations sequentially
        experts = []
        current_input = x
        for i in range(3):
            expert_output = self.altup_proj[i](current_input)
            expert_output = torch.clamp(expert_output, -10, 10)  # Gradient clipping
            experts.append(expert_output)
            current_input = expert_output  # Chain experts together
            
        # Mix original input and expert outputs based on routing probabilities
        all_outputs = torch.stack([x] + experts, dim=-1)  # [B, T, D, 4]
        mixed = torch.sum(all_outputs * router_probs.unsqueeze(-2), dim=-1)
        
        return mixed
        
    def altup_correct(self, predicted, actual):
        """ALTUP correction step: correct prediction based on actual computation"""
        # Compute prediction error
        error = actual - predicted
        
        # Get correction routing weights
        router_weights = self.altup_router_correct(actual)
        # Add bias to correction routing for stability
        router_probs = F.softmax(router_weights + 0.5, dim=-1)
        
        # Apply corrections through unprojection experts
        corrections = []
        for i in range(3):
            correction = self.altup_unproj[i](error)
            correction = torch.clamp(correction, -10, 10)
            corrections.append(correction)
            
        # Weighted combination of corrections
        corrections_stack = torch.stack(corrections, dim=-1)  # [B, T, D, 3]
        # Use first 3 routing probabilities for corrections
        correction_weights = router_probs[:, :, 1:4]  # Skip first weight (identity)
        weighted_correction = torch.sum(corrections_stack * correction_weights.unsqueeze(-2), dim=-1)
        
        return predicted + weighted_correction
    
    def forward(self, x, position_ids, per_layer_emb):
        # ALTUP prediction: predict what the layer output should be  predicted.shape= [B, T, D]
        predicted = self.altup_predict(x)
        
        # Standard transformer computation
        # Attention block with pre/post normalization 
        h = self.pre_attn_norm(predicted) #shape= [B, T, D]
        h = self.attention(h, position_ids) 
        h = self.post_attn_norm(h)
        h = predicted + h  # Residual connection
        
        # Laurel residual block for additional skip connections
        h = self.laurel(h)
        h = h * self.residual_scale  # Scale residual for stability
        
        # MLP block with pre/post normalization
        h_norm = self.pre_mlp_norm(h)
        mlp_out = self.mlp(h_norm)
        mlp_out = self.post_mlp_norm(mlp_out)
        h = h + mlp_out  # Residual connection
        
        # ALTUP correction: correct prediction based on actual computation B T D , B T D
        corrected = self.altup_correct(predicted, h)
        
        # Integrate per-layer embeddings
        gate = F.gelu(self.per_layer_gate(corrected)) # B T D ==> [B, T, 256]
        per_layer_contribution = gate * per_layer_emb[:, :, self.layer_idx] # [B, T, 256]
        per_layer_out = self.per_layer_proj(per_layer_contribution) # [B, T, D]
        per_layer_out = self.per_layer_norm(per_layer_out)
        
        return corrected + per_layer_out
    

class GeminiModel(nn.Module):	
    """Complete Gemini model with ALTUP layers and per-layer embeddings"""	
    def __init__(self, vocab_size=262144, dim=2048, num_layers=30):	
        super().__init__()	
        self.vocab_size = vocab_size	
        self.dim = dim	
        self.num_layers = num_layers	

        # Per-layer embedding processing
        per_layer_total_dim = num_layers * 256
        self.per_layer_proj = nn.Linear(per_layer_total_dim, per_layer_total_dim, bias=False)	
        self.per_layer_norm = LayerNorm(256)	

        # Stack of transformer layers with ALTUP
        self.layers = nn.ModuleList([	
            TransformerLayer(dim, i) for i in range(num_layers)	
        ])	

        # Final output processing
        self.final_norm = LayerNorm(dim)	
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)	

        # Output scaling for numerical stability
        self.output_scale = 30.0

    def forward(self, embeddings, per_layer_embeddings, position_ids):	
        # Process per-layer embeddings
        B, T, num_layers, emb_dim = per_layer_embeddings.shape	
        
        # Flatten per-layer embeddings for processing
        per_layer_flat = per_layer_embeddings.view(B, T, -1)	
        per_layer_processed = self.per_layer_proj(per_layer_flat)	
        per_layer_reshaped = per_layer_processed.view(B, T, num_layers, emb_dim)	
        
        # Normalize and add residual connection
        per_layer_normed = self.per_layer_norm(per_layer_reshaped)	
        per_layer_emb = per_layer_normed + per_layer_embeddings	# Residual connection
        per_layer_emb = per_layer_emb * 0.167  # Scaling factor for stability

        # Forward pass through transformer layers
        h = embeddings	        
        for layer in self.layers:	        
            h = layer(h, position_ids, per_layer_emb)

        # Final output projection with scaling
        h = self.final_norm(h)	
        logits = self.lm_head(h)	
        
        # Apply output scaling with tanh clamping
        logits = logits / self.output_scale	
        logits = torch.tanh(logits) * self.output_scale	

        return logits	

class PerLayerEmbedder(nn.Module):	
    """Per-layer embedding lookup for each transformer layer"""	
    def __init__(self, vocab_size=262144, embedding_dim=256, num_layers=30):	
        super().__init__()	
        self.num_layers = num_layers	
        self.vocab_size = vocab_size
        # Separate embedding table for each layer
        self.embeddings = nn.ModuleList([	
            nn.Embedding(vocab_size, embedding_dim) for _ in range(num_layers)	
        ])	

    def forward(self, token_ids):	
        # Ensure token IDs are within valid range
        token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)

        # Get embeddings from each layer's embedding table
        layer_embeddings = []	
        for emb in self.embeddings:	
            layer_embeddings.append(emb(token_ids))	

        # Stack to [batch, seq_len, num_layers, embedding_dim]	
        return torch.stack(layer_embeddings, dim=2)	

# Example usage and testing
if __name__ == "__main__":	
    batch_size = 2	
    seq_len = 8	

    # Initialize models	
    embedder = PerLayerEmbedder()	
    model = GeminiModel()	

    # Create example inputs	
    token_ids = torch.randint(0, 262144, (batch_size, seq_len))	
    embeddings = torch.randn(batch_size, seq_len, 2048)	
    position_ids = torch.arange(seq_len)  # Keep it 1D, RoPE will handle batching

    # Get per-layer embeddings	
    per_layer_emb = embedder(token_ids)	
    print(f"Per-layer embeddings shape: {per_layer_emb.shape}")

    # Run model	
    logits = model(embeddings, per_layer_emb, position_ids)	
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")