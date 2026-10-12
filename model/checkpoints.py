"""Strict, configuration-driven loading of PRISM encoder checkpoints."""

import torch

from model.mpn_vas_st import PRISMPriorEncoder


def load_encoder(path, sample, device, allow_global_features=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("encoder_state_dict", checkpoint)
    config = checkpoint.get("encoder_config")
    if config is None:
        # Recover architecture from older PRISM state dictionaries where possible.
        old = checkpoint.get("config", {})
        global_weight = state.get("global_encoder.weight")
        config = {
            "num_node_features": int(sample.x.shape[1]),
            "num_edge_features": int(sample.edge_attr.shape[1]),
            "num_global_features": 0 if global_weight is None else int(global_weight.shape[1]),
            "hidden_dim": int(state["mpn.encoder.W_i.weight"].shape[0]),
            "projection_dim": int(state["projection_head.3.weight"].shape[0]),
            "cov_num": old.get("cov_num") or 3,
            "dropout_rate": old.get("dropout_rate", 0.3),
            "property_index": old.get("property_index", 0),
            "tau": old.get("tau", 0.1),
            "tau_phys_scale": old.get("tau_phys_scale", 0.5),
            "atom_messages": old.get("atom_messages", True),
        }
    else:
        config = dict(config)
    for name, actual in (("num_node_features", sample.x.shape[1]), ("num_edge_features", sample.edge_attr.shape[1])):
        if config[name] != actual:
            raise ValueError(f"{path}: {name}={config[name]} does not match graph features ({actual}).")
    if config.get("num_global_features", 0) and not allow_global_features:
        raise ValueError(
            f"{path} was pretrained with global features. Use a checkpoint trained without them "
            "for the default protocol, or explicitly pass --use_global_features."
        )
    encoder = PRISMPriorEncoder(**config).to(device)
    encoder.load_state_dict(state, strict=True)
    return encoder


def load_encoders(sample, paths, device, allow_global_features=False):
    encoders = [load_encoder(p, sample, device, allow_global_features) for p in paths]
    encoder_output_dim(encoders)
    return encoders


def encoder_output_dim(encoders):
    dims = {encoder.output_dim for encoder in encoders}
    if len(dims) != 1:
        raise ValueError(f"PRISM fusion requires equal branch dimensions, received {sorted(dims)}.")
    return dims.pop()
