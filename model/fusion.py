import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticWeightedFusion(nn.Module):
    """
    Static weighted fusion over prior-specific branches.

    Input:
        embs: [B, K, D]
        global_features: optional [B, G_in] or [B, 1, G_in]

    If pair=False:
        return prediction, weights
    If pair=True:
        return fused embedding, weights
    """

    def __init__(
        self,
        num_inputs=4,
        emb_dim=512,
        dropout=0.1,
        layer_norm_out=True,
        global_dim=32,
        global_in_dim=8,
        pair=False,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.emb_dim = emb_dim
        self.pair = pair

        self.branch_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Global, sample-independent branch weights.
        self.weight_logits = nn.Parameter(torch.zeros(num_inputs))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim) if layer_norm_out else None

        self.global_encoder = nn.Linear(global_in_dim, global_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + global_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def _encode_global(self, global_features, B, device):
        if global_features is None:
            global_features = torch.zeros(
                B,
                self.global_encoder.in_features,
                device=device,
            )
        else:
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)

            if global_features.dim() == 3 and global_features.size(1) == 1:
                global_features = global_features.squeeze(1)

            if global_features.size(0) == 1 and B > 1:
                global_features = global_features.expand(B, -1)

            assert global_features.size(0) == B, (
                f"global_features batch mismatch: "
                f"got {global_features.size(0)}, expected {B}"
            )

        return self.global_encoder(global_features)

    def forward(self, embs, global_features=None):
        assert embs.dim() == 3, f"embs must be [B, K, D], got {tuple(embs.shape)}"

        B, K, D = embs.size()
        assert K == self.num_inputs, (
            f"num_inputs mismatch: embs K={K}, "
            f"but self.num_inputs={self.num_inputs}"
        )

        x = self.branch_proj(embs)  # [B, K, D]

        weights = torch.softmax(self.weight_logits, dim=0)  # [K]
        fused = torch.einsum("bkd,k->bd", x, weights)       # [B, D]

        fused = self.dropout(fused)
        if self.layer_norm is not None:
            fused = self.layer_norm(fused)

        if self.pair:
            return fused, weights

        g = self._encode_global(global_features, B, embs.device)
        fused_plus_g = torch.cat([fused, g], dim=-1)

        out = self.mlp(fused_plus_g).squeeze(-1)
        return out, weights


class AdaptiveWeightedFusion(nn.Module):
    """
    Input-adaptive weighted fusion over prior-specific branches.

    The branch weights are predicted for each sample.
    Optionally, molecular global features can participate in the gating network.

    Input:
        embs: [B, K, D]
        global_features: optional [B, G_in] or [B, 1, G_in]

    If pair=False:
        return prediction, weights
    If pair=True:
        return fused embedding, weights
    """

    def __init__(
        self,
        num_inputs=4,
        emb_dim=512,
        dropout=0.1,
        layer_norm_out=True,
        global_dim=32,
        global_in_dim=8,
        use_global_in_weights=True,
        pair=False,
        temperature_init=1.0,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.emb_dim = emb_dim
        self.pair = pair
        self.use_global_in_weights = use_global_in_weights

        self.branch_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.global_encoder = nn.Linear(global_in_dim, global_dim)

        gate_in_dim = emb_dim + (global_dim if use_global_in_weights else 0)
        self.weight_net = nn.Sequential(
            nn.Linear(gate_in_dim, max(128, emb_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(128, emb_dim // 2), 1),
        )

        # Sample-independent prior-space bias.
        self.weight_bias = nn.Parameter(torch.zeros(num_inputs))

        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim) if layer_norm_out else None

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + global_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def _encode_global(self, global_features, B, device):
        if global_features is None:
            global_features = torch.zeros(
                B,
                self.global_encoder.in_features,
                device=device,
            )
        else:
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)

            if global_features.dim() == 3 and global_features.size(1) == 1:
                global_features = global_features.squeeze(1)

            if global_features.size(0) == 1 and B > 1:
                global_features = global_features.expand(B, -1)

            assert global_features.size(0) == B, (
                f"global_features batch mismatch: "
                f"got {global_features.size(0)}, expected {B}"
            )

        return self.global_encoder(global_features)

    def forward(self, embs, global_features=None):
        assert embs.dim() == 3, f"embs must be [B, K, D], got {tuple(embs.shape)}"

        B, K, D = embs.size()
        assert K == self.num_inputs, (
            f"num_inputs mismatch: embs K={K}, "
            f"but self.num_inputs={self.num_inputs}"
        )

        x = self.branch_proj(embs)  # [B, K, D]
        g = self._encode_global(global_features, B, embs.device)  # [B, global_dim]

        if self.use_global_in_weights:
            g_expand = g.unsqueeze(1).expand(-1, K, -1)
            gate_input = torch.cat([x, g_expand], dim=-1)  # [B, K, D + global_dim]
        else:
            gate_input = x

        dynamic_logits = self.weight_net(gate_input).squeeze(-1)  # [B, K]
        logits = dynamic_logits + self.weight_bias.unsqueeze(0)

        temperature = torch.clamp(self.temperature, min=1e-2)
        weights = F.softmax(logits / temperature, dim=1)  # [B, K]

        fused = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, D]

        fused = self.dropout(fused)
        if self.layer_norm is not None:
            fused = self.layer_norm(fused)

        if self.pair:
            return fused, weights

        fused_plus_g = torch.cat([fused, g], dim=-1)
        out = self.mlp(fused_plus_g).squeeze(-1)

        return out, weights


class FusionFineTuneModel(nn.Module):
    """
    Fine-tuning wrapper for ChemSpaceA / PRISM branch encoders.

    fusion_type:
        - "static": sample-independent global branch weights
        - "adaptive": sample-dependent branch weights
    """

    def __init__(
        self,
        encoder_list,
        fusion_model,
        fusion_type="adaptive",
    ):
        super().__init__()
        assert fusion_type in {"static", "adaptive"}, (
            "fusion_type must be either 'static' or 'adaptive'"
        )

        self.encoders = nn.ModuleList(encoder_list)
        self.fusion = fusion_model
        self.fusion_type = fusion_type

    def forward(self, data):
        embs = [encoder(data) for encoder in self.encoders]
        embs = torch.stack(embs, dim=1)  # [B, K, D]

        global_features = getattr(data, "global_features", None)

        out, weights = self.fusion(embs, global_features)

        return out, weights

WeightedFusion = StaticWeightedFusion
WeightedFusion_new = AdaptiveWeightedFusion