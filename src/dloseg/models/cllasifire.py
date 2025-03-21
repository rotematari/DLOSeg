import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskClassifierNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(MaskClassifierNetwork, self).__init__()

        # Projection MLP to transform prompt embeddings
        self.proj_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.cross_norm = nn.LayerNorm(256)

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.self_norm = nn.LayerNorm(256)

        # Final MLP classifier
        self.classifier_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, point_prompt_embeddings, mask_tokens):
        # point_prompt_embeddings: (B, N*Np, 256)
        # mask_tokens: (B, N, 256)

        # Transform embeddings through MLP
        queries = self.project_embed(point_prompt_embeddings)

        # Cross-attention: Queries (prompt embeddings) attend to mask_tokens
        cross_attn_out, _ = self.cross_attention(query=point_prompt_embeddings,
                                                 key=mask_tokens,
                                                 value=mask_tokens)

        cross_attn_out = self.cross_norm(cross_attn_out + point_prompt_embeddings)

        # Self-attention to refine embeddings
        self_attn_out, _ = self.cross_attention(query=cross_attn_out,
                                                key=cross_attn_out,
                                                value=cross_attn_out)

        self_attn_out = self.cross_norm(self_attn_out + cross_attn_out)

        # Classifier MLP
        logits = self.classifier_mlp(self_attn_out).squeeze(-1)
        binary_preds = torch.sigmoid(logits)

        return binary_preds