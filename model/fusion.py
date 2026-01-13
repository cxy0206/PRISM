import torch
import torch.nn as nn
import math

class TransformerFusionModel(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512,global_dim=32, pair=False):
        super().__init__()
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.query  = nn.Parameter(torch.randn(emb_dim))
        self.global_encoder = nn.Linear(8, global_dim) 
        self.pair = pair
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+global_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x,global_features, pair=False):
        B, N, D = x.size()
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = self.query.unsqueeze(0).unsqueeze(1).expand(B, 1, D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        weights = torch.softmax(scores, dim=-1)
        fused = torch.matmul(weights, V).squeeze(1)
        global_features = global_features.squeeze(1)
        fused = torch.cat([fused, self.global_encoder(global_features)], dim=-1)  # [B, D + G]
        out = self.mlp(fused).squeeze(-1)
        if not pair:
            return out, weights.squeeze(1)
        else :
            return fused, weights.squeeze(1)


class WeightedFusion(nn.Module):
    def __init__(self, num_inputs=3, emb_dim=512, dropout=0.1, layer_norm_out=True, global_dim=32,pair=False):
    
        super().__init__()
        self.emb_dim = emb_dim
        self.num_inputs = num_inputs
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.weight_logits = nn.Parameter(torch.zeros(num_inputs))  # initialized to uniform weights
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim) if layer_norm_out else None
        self.global_encoder = nn.Linear(8, global_dim)  # Assuming global features are of size 8
        self.pair = pair
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+global_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, embs, global_features):  # embs: [B, N, D]
        B, N, D = embs.size()
        x = self.linear(embs)  # shape [B, N, D]
        norm_weights = torch.softmax(self.weight_logits, dim=0)  # shape [N]
        fused = torch.einsum('bnd,n->bd', x, norm_weights)  # [B, D]
        fused = self.dropout(fused)
        if self.layer_norm is not None:
            fused = self.layer_norm(fused)

        global_features = global_features.squeeze(1)
        # print("Fused features shape:", fused.shape)
        # print("Global features shape:", global_features.shape)
        fused = torch.cat([fused, self.global_encoder(global_features)], dim=-1)  # [B, D + G]

        out = self.mlp(fused).squeeze(-1)
        if not self.pair:
            return out, norm_weights
        else:
            return fused, norm_weights

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFusion_new(nn.Module):
    def __init__(
        self,
        num_inputs=3,
        emb_dim=512,
        dropout=0.1,
        layer_norm_out=True,
        global_dim=32,
        global_in_dim=8,          #  新增：全局特征原始维度（默认为 8）
        use_global_in_weights=True, #  新增：是否让全局特征参与权重计算
        pair=False,
        temperature_init=1.0      #  可选：softmax 温度
    ):
        super().__init__()
        self.emb_dim   = emb_dim
        self.num_inputs = num_inputs
        self.pair = pair
        self.use_global_in_weights = use_global_in_weights

        # 对每个分支的特征先做一层“投影/非线性”
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        #  全局先验：对 N 个分支的可学习偏置（与样本无关）
        self.weight_bias = nn.Parameter(torch.zeros(num_inputs))

        #  用于“逐样本”的权重 logits 预测：输入是 [x_i ; g]，输出是标量 logit
        gate_in_dim = emb_dim + (global_dim if use_global_in_weights else 0)
        self.weight_net = nn.Sequential(
            nn.Linear(gate_in_dim, max(128, emb_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(128, emb_dim // 2), 1)  # -> per-branch logit
        )

        # 输出端的正则化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim) if layer_norm_out else None

        # 全局特征编码器（用于最终预测头 & 可选用于权重门控）
        self.global_encoder = nn.Linear(global_in_dim, global_dim)

        # 预测头（在融合后的向量与全局向量拼接上做回归）
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
            nn.Linear(64, 1)
        )

        #  温度参数（可学习或固定，这里给个可学习版本更灵活）
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))

    def _encode_global(self, global_features, B: int, device):
        if global_features is None:
            gf = torch.zeros(B, self.global_encoder.in_features, device=device)
        else:
            gf = global_features
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)  # [1,8]
            if gf.dim() == 3 and gf.size(1) == 1:
                gf = gf.squeeze(1)    # [B,8]
            # 若维度仍不为 [B,8]，尝试广播到 B
            if gf.size(0) == 1 and B > 1:
                gf = gf.expand(B, -1)
            assert gf.shape[0] == B, f"global_features batch mismatch: got {gf.shape[0]}, expect {B}"
        return self.global_encoder(gf)  # -> [B, global_dim]

    def forward(self, embs, global_features=None):
    
        assert embs.dim() == 3, f"embs must be [B,N,D], got {tuple(embs.shape)}"
        B, N, D = embs.size()
        assert N == self.num_inputs, f"num_inputs mismatch: embs N={N} vs self.num_inputs={self.num_inputs}"

        x = self.linear(embs)      # [B, N, D]

        g = self._encode_global(global_features, B, embs.device)  # [B, global_dim]

        if self.use_global_in_weights:
            g_expand = g.unsqueeze(1).expand(-1, N, -1)
            gate_in = torch.cat([x, g_expand], dim=-1)  # [B, N, D+G]
        else:
            gate_in = x  
        dyn_logits = self.weight_net(gate_in).squeeze(-1)        # [B, N]
        logits = dyn_logits + self.weight_bias.unsqueeze(0)      # [B, N]

        temp = torch.clamp(self.temperature, min=1e-2)
        norm_weights = F.softmax(logits / temp, dim=1)           # [B, N]

        fused = torch.sum(x * norm_weights.unsqueeze(-1), dim=1) # [B, D]
        fused = self.dropout(fused)
        if self.layer_norm is not None:
            fused = self.layer_norm(fused)

        fused_plus_g = torch.cat([fused, g], dim=-1)             # [B, D+G]
        out = self.mlp(fused_plus_g).squeeze(-1)                 # [B]

        if not self.pair:
            return out, norm_weights
        else:
            return fused, norm_weights

class MLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim=64,global_dim=32,pair=False):
        super().__init__()
        self.global_encoder = nn.Linear(8, global_dim)  # Assuming global features are of size 8
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+global_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.pair = pair

    def forward(self, x, global_features):
        global_features = global_features.squeeze(1)
        x = torch.cat([x, self.global_encoder(global_features)], dim=-1)  # [B, D + G]
        out = self.mlp(x).squeeze(-1)
        if not self.pair:
            return out
        return torch.cat([x, self.global_encoder(global_features)], dim=-1)


# fine-tune model
class FusionFineTuneModel(nn.Module):
    def __init__(self, encoder_list, fusion_model, fusion_method='attention'):
        super().__init__()
        self.encoders = nn.ModuleList(encoder_list)
        self.fusion = fusion_model
        self.fusion_method = fusion_method

    def forward(self, data):
        embs = [encoder(data) for encoder in self.encoders]  # list of [B, D]
        embs = torch.stack(embs, dim=1)  # [B, 3, D]
        if self.fusion_method == 'attention':
            out, weights = self.fusion(embs,data.global_features)
            return out, weights
        else:
            weights = 0
            out = self.fusion(torch.cat([embs[:, i, :] for i in range(embs.size(1))], dim=1), data.global_features)
            return out, weights
