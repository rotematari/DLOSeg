import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sam2_rt.sam2.build_sam import build_sam2
from src.sam2_rt.sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)


# ------------------------------
# 2. Prompt Encoder Network
# ------------------------------
class PromptEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_prompts, points_per_prompt, freq_matrix):
        """
        Args:
            input_dim (int): Dimension of the upsampled attention embedding from CLIPSeg.
            embed_dim (int): Model embedding dimension (e.g., 256).
            num_prompts (int): N, the number of prompt batches.
            points_per_prompt (int): Np, the number of points per prompt.
            freq_matrix (Tensor): Frequency matrix for DPE.
        """
        super().__init__()
        self.num_prompts = num_prompts
        self.points_per_prompt = points_per_prompt
        self.embed_dim = embed_dim

        # Project CLIPSeg embedding into our working dimension.
        self.proj = nn.Linear(input_dim, embed_dim)
        self.dpe = DensePositionalEncoding(embed_dim, freq_matrix)

        # Single self-attention layer.
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

        # MLP to filter/select viable query patches.
        # Here we simply aggregate (via mean) the self-attended patches,
        # then output num_prompts*points_per_prompt embeddings.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_prompts * points_per_prompt * embed_dim)
        )
        # A linear layer to mimic SAM’s labeling protocol.
        self.label_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, attn_embedding):
        """
        Args:
            attn_embedding (Tensor): Upsampled attention embedding from CLIPSeg.
                                     Shape: (B, H, W, input_dim)
        Returns:
            point_prompts: Tensor of shape (B, num_prompts, points_per_prompt, embed_dim)
        """
        B, H, W, _ = attn_embedding.shape
        # Project and add positional encoding.
        x = self.proj(attn_embedding)  # (B, H, W, embed_dim)
        pos_enc = self.dpe((H, W)).to(x.device)  # (H, W, embed_dim)
        x = x + pos_enc  # broadcast to (B, H, W, embed_dim)

        # Flatten spatial dimensions to sequence: (B, H*W, embed_dim)
        x_flat = x.view(B, H * W, self.embed_dim)
        # Self-attention over patches.
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        attn_out = self.ln(attn_out + x_flat)

        # Aggregate information (here, via global average pooling)
        pooled = attn_out.mean(dim=1)  # (B, embed_dim)
        # Generate queries for prompts: (B, num_prompts*points_per_prompt*embed_dim)
        queries = self.mlp(pooled)
        # Reshape to (B, num_prompts, points_per_prompt, embed_dim)
        queries = queries.view(B, self.num_prompts, self.points_per_prompt, self.embed_dim)
        # Apply a final linear transformation per point.
        point_prompts = self.label_linear(queries)
        return point_prompts

# ------------------------------
# 3. Mask Classification Network
# ------------------------------
class MaskClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes=2):
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_classes (int): Number of classes (binary: foreground/background).
        """
        super().__init__()
        # Projection MLP for point embeddings.
        self.proj_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Cross-attention: point embeddings (queries) attend to SAM mask tokens (keys/values).
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # Self-attention block for classifier tokens.
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        # Final MLP for binary classification.
        self.classifier_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, point_embeddings, mask_tokens, text_features=None):
        """
        Args:
            point_embeddings (Tensor): From the prompt encoder.
                                         Shape: (B, num_prompts, points_per_prompt, embed_dim)
            mask_tokens (Tensor): Mask tokens from SAM’s decoder.
                                  Shape: (B, num_masks, embed_dim)
            text_features (Tensor, optional): Text-conditioned features.
        Returns:
            logits: Binary classification logits,
                    shape: (B, num_prompts, points_per_prompt, num_classes)
        """
        B, num_prompts, points_per_prompt, embed_dim = point_embeddings.shape
        # Flatten point embeddings to a sequence: (B, num_prompts * points_per_prompt, embed_dim)
        queries = point_embeddings.view(B, num_prompts * points_per_prompt, embed_dim)
        queries = self.proj_mlp(queries)
        # Cross-attention: queries attend to mask tokens.
        cross_out, _ = self.cross_attn(queries, mask_tokens, mask_tokens)
        cross_out = self.ln(cross_out + queries)  # Residual connection

        # Optionally, fuse in text features (e.g. via addition of a global text summary).
        if text_features is not None:
            text_summary = text_features.mean(dim=1, keepdim=True)  # (B, 1, embed_dim)
            cross_out = cross_out + text_summary

        # Self-attention over the combined features.
        self_attn_out, _ = self.self_attn(cross_out, cross_out, cross_out)
        self_attn_out = self.ln(self_attn_out + cross_out)

        # Classify each point prompt.
        logits = self.classifier_mlp(self_attn_out)  # (B, num_prompts*points_per_prompt, num_classes)
        logits = logits.view(B, num_prompts, points_per_prompt, -1)
        return logits

# ------------------------------
# 4. Adapter Module Combining Both Networks
# ------------------------------
class Adapter(nn.Module):
    def __init__(self, attn_input_dim, embed_dim, num_prompts, points_per_prompt, freq_matrix, num_classes=2):
        """
        Args:
            attn_input_dim (int): Dimension of the CLIPSeg attention embedding.
            embed_dim (int): Common embedding dimension (e.g., 256).
            num_prompts (int): N, number of prompt batches.
            points_per_prompt (int): Np, number of points in each prompt.
            freq_matrix (Tensor): Frequency matrix for DPE.
            num_classes (int): Number of classification classes.
        """
        super().__init__()
        self.prompt_encoder = PromptEncoder(attn_input_dim, embed_dim, num_prompts, points_per_prompt, freq_matrix)
        self.mask_classifier = MaskClassifier(embed_dim, num_classes)

    def forward(self, attn_embedding, mask_tokens, text_features=None):
        """
        Args:
            attn_embedding (Tensor): Upsampled CLIPSeg attention embedding,
                                     shape (B, H, W, attn_input_dim).
            mask_tokens (Tensor): SAM mask tokens, shape (B, num_masks, embed_dim).
            text_features (Tensor, optional): Text embeddings if available.
        Returns:
            logits: Classification logits for each prompt point.
            point_prompts: The generated prompt embeddings.
        """
        point_prompts = self.prompt_encoder(attn_embedding)
        logits = self.mask_classifier(point_prompts, mask_tokens, text_features)
        return logits, point_prompts

# ------------------------------
# Example usage:
# ------------------------------
if __name__ == "__main__":
    # Dummy dimensions and frequency matrix.
    B, H, W = 2, 16, 16
    attn_input_dim = 512    # Example dimension from CLIPSeg's attention map.
    embed_dim = 256
    num_prompts = 4
    points_per_prompt = 10
    num_masks = 8
    num_classes = 2

    # Create a fixed frequency matrix (e.g., 32 frequencies)
    K = 32
    freq_matrix = torch.randn(K, 2)

    # Dummy inputs.
    attn_embedding = torch.randn(B, H, W, attn_input_dim)
    mask_tokens = torch.randn(B, num_masks, embed_dim)
    text_features = torch.randn(B, 77, embed_dim)  # e.g., from a CLIP text encoder

    # Instantiate adapter.
    adapter = Adapter(attn_input_dim, embed_dim, num_prompts, points_per_prompt, freq_matrix, num_classes)
    logits, prompts = adapter(attn_embedding, mask_tokens, text_features)

    print("Logits shape:", logits.shape)      # Expected: (B, num_prompts, points_per_prompt, num_classes)
    print("Prompts shape:", prompts.shape)      # Expected: (B, num_prompts, points_per_prompt, embed_dim)
