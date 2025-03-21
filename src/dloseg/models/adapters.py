import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2_realtime.sam2.modeling.position_encoding import PositionEmbeddingRandom

class PromptEncoder(nn.Module):
    def __init__(self,configs):
        super().__init__()
        # hiperparameters
        self.N = configs.get("N", 1) # Number of prompts
        self.Np = configs.get("Np", 3) # Number of points per prompt
        self.embedding_dim = configs.get("embedding_dim", 256) # Embedding dimension
        self.attention_dropout = configs.get("attention_dropout", 0.1) # Dropout for attention layers
        self.attention_heads = configs.get("attention_heads", 8) # Number of attention heads
        self.filtering_mlp_dim = configs.get("filtering_mlp_dim", 1024) # Dimension of filtering MLP
        

        # Linear layer to project CLIPSeg embeddings to required dimension
        self.project_embed = nn.Linear(64, self.embedding_dim)

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim,
                                               num_heads=self.attention_heads,
                                               batch_first=True,
                                               dropout=self.attention_dropout)
        self.attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        
        self.filtering_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.filtering_mlp_dim),
            nn.GELU(),
            nn.Linear(self.filtering_mlp_dim,self.embedding_dim),
        )
        self.filtering_norm = nn.LayerNorm(self.embedding_dim)
        
        self.sampler_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim,
                                                  num_heads=self.attention_heads,
                                                  batch_first=True,
                                                  dropout=self.attention_dropout)
        
        self.sampler_norm = nn.LayerNorm(self.embedding_dim)
        
        self.label_layer = nn.Linear(self.embedding_dim, 3)
        self.label_norm = nn.LayerNorm(3)
        

    def forward(self, clipseg_embedding, dpe):
        """
        clipseg_embedding: [B, H, W, 64] tensor from CLIPSeg embeddings.
        dpe: [B, H, W, 256] Dense Positional Encoding tensor
        """
        B, H, W, C = clipseg_embedding.shape

        # Project embeddings to SAM-compatible dimension
        clipseg_proj = self.project_embed(clipseg_embedding)  # (B, H, W, embedding_dim)

        # Combine embeddings and positional encodings
        embeddings = clipseg_proj + dpe  # (B, H, W, embedding_dim)

        # Flatten to sequence form for self-attention
        embeddings_flat = embeddings.view(B, H*W, -1) # (B, H*W, embedding_dim)

        # Apply self-attention
        embeddings_attn, _ = self.self_attn(embeddings_flat, embeddings_flat, embeddings_flat)
        embeddings_attn = self.attn_layer_norm(embeddings_attn) # (B, H*W, embedding_dim)

        # Add positional encodings back
        embeddings_attn += dpe.view(B, H*W, -1)
        # Filtering MLP
        embeddings_filtered = self.filtering_mlp(embeddings_attn)
        embeddings_filtered = self.filtering_norm(embeddings_filtered) # (B, H*W, embedding_dim)
        
        # Sample attention
        embeddings_sampled, _ = self.sampler_attn(query = embeddings_filtered,
                                                  key = embeddings_attn,
                                                  value = dpe.view(B, H*W, -1))
        embeddings_sampled = self.sampler_norm(embeddings_sampled) # (B, H*W, embedding_dim)
        
        # Classify points: foreground/background/no-point
        point_class_logits = self.label_layer(embeddings_sampled)
        point_embeddings = self.label_norm(point_class_logits) # (B, H*W, 3)


        return point_embeddings, embeddings_sampled

class Classifier(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.embedding_dim = configs.get("embedding_dim", 256)
        self.attention_dropout = configs.get("attention_dropout", 0.1)
        self.attention_heads = configs.get("attention_heads", 8)
        self.mlp_dim = configs.get("classifier_mlp_dim", 1024)

        # MLP to transform sampled point embeddings
        self.input_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )

        # Cross-attention block
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(self.embedding_dim)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(self.embedding_dim)

        # Final MLP for binary classification
        self.output_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim, 2)
        )

    def forward(self, embeddings_sampled, mask_tokens):
        """
        Forward pass for the classifier module.
        
        Args:
            point_embeddings: [B, H*W, 3] tensor of sampled point embeddings
            mask_tokens: [B, H , W, embedding_dim] tensor of mask tokens
            
        Returns:
            logits: [B, 2] tensor with binary classification logits
        """
        x = self.input_mlp(embeddings_sampled)

        # Cross-attention
        attn_output, _ = self.cross_attention(
            query=x,
            key=mask_tokens,
            value=mask_tokens
        )
        x = self.cross_attn_norm(x + attn_output)

        # Self-attention
        self_attn_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x
        )
        x = self.self_attn_norm(x + self_attn_output)

        # Mean pooling and classification
        x = torch.mean(x, dim=1)
        logits = self.output_mlp(x)

        return logits

