import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import BertConfig


def msign(G, steps=5, eps=1e-7):
    """Approximate the orthogonal sign matrix used by spikingbert_rot.py."""
    a, b, c = (3.4445, -4.7750, 2.0315)

    if G.dim() == 2:
        G = G.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    norm = G.norm(dim=(-2, -1), keepdim=True)
    X = G / (norm + eps)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if squeeze_output:
        X = X.squeeze(0)

    return X


def fuse_rotation_matrices_in_state_dict(state_dict, config):
    """Fuse per-layer rotations into linear weights.

    The global bert.H1 is intentionally kept. spikingbert_rot_inf.py still
    applies the embedding-side H1 and final encoder-output H1.T at runtime.
    Only per-layer H2/H3 matrices are removed.
    """
    state_dict = deepcopy(state_dict)

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    mlp_ratio = intermediate_size // hidden_size

    original_dtypes = {key: value.dtype for key, value in state_dict.items() if torch.is_floating_point(value)}
    H1 = msign(state_dict["bert.H1"].to(torch.float64))

    for layer_idx in range(num_layers):
        prefix = f"bert.encoder.layer.{layer_idx}"

        H2_key = f"{prefix}.attention.H2"
        H3_key = f"{prefix}.H3"
        H2 = msign(state_dict[H2_key].to(torch.float64))
        H3 = msign(state_dict[H3_key].to(torch.float64))

        q_weight_key = f"{prefix}.attention.self.query.weight"
        k_weight_key = f"{prefix}.attention.self.key.weight"
        state_dict[q_weight_key] = state_dict[q_weight_key].to(torch.float64) @ H1
        state_dict[k_weight_key] = state_dict[k_weight_key].to(torch.float64) @ H1

        v_weight_key = f"{prefix}.attention.self.value.weight"
        v_bias_key = f"{prefix}.attention.self.value.bias"
        state_dict[v_weight_key] = H2.T @ state_dict[v_weight_key].to(torch.float64) @ H1
        if v_bias_key in state_dict:
            state_dict[v_bias_key] = H2.T @ state_dict[v_bias_key].to(torch.float64)

        attn_dense_key = f"{prefix}.attention.output.dense.weight"
        attn_bias_key = f"{prefix}.attention.output.dense.bias"
        state_dict[attn_dense_key] = H1.T @ state_dict[attn_dense_key].to(torch.float64) @ H2
        if attn_bias_key in state_dict:
            state_dict[attn_bias_key] = H1.T @ state_dict[attn_bias_key].to(torch.float64)

        inter_dense_key = f"{prefix}.intermediate.dense.weight"
        inter_bias_key = f"{prefix}.intermediate.dense.bias"
        inter_weight = state_dict[inter_dense_key].to(torch.float64).view(mlp_ratio, hidden_size, hidden_size)
        inter_weight = torch.matmul(H3.permute(0, 2, 1), inter_weight)
        state_dict[inter_dense_key] = inter_weight.reshape(intermediate_size, hidden_size) @ H1
        if inter_bias_key in state_dict:
            inter_bias = state_dict[inter_bias_key].to(torch.float64).view(mlp_ratio, hidden_size, 1)
            state_dict[inter_bias_key] = torch.matmul(H3.permute(0, 2, 1), inter_bias).reshape(-1)

        out_dense_key = f"{prefix}.output.dense.weight"
        out_bias_key = f"{prefix}.output.dense.bias"
        out_weight = state_dict[out_dense_key].to(torch.float64).view(hidden_size, mlp_ratio, hidden_size).permute(1, 0, 2)
        out_weight = torch.matmul(out_weight, H3)
        state_dict[out_dense_key] = H1.T @ out_weight.permute(1, 0, 2).reshape(hidden_size, intermediate_size)
        if out_bias_key in state_dict:
            state_dict[out_bias_key] = H1.T @ state_dict[out_bias_key].to(torch.float64)

        del state_dict[H2_key]
        del state_dict[H3_key]

    for key, dtype in original_dtypes.items():
        if key in state_dict:
            state_dict[key] = state_dict[key].to(dtype)

    return state_dict


def _resolve_weight_file(path):
    path = Path(path)
    if path.is_dir():
        safetensors_path = path / "model.safetensors"
        bin_path = path / "pytorch_model.bin"
        if safetensors_path.exists():
            return safetensors_path
        if bin_path.exists():
            return bin_path
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {path}")
    return path


def load_state_dict(path):
    weight_file = _resolve_weight_file(path)
    if weight_file.suffix == ".safetensors":
        return load_file(str(weight_file), device="cpu"), weight_file

    loaded = torch.load(weight_file, map_location="cpu")
    if "model_state_dict" in loaded:
        loaded = loaded["model_state_dict"]
    elif "state_dict" in loaded:
        loaded = loaded["state_dict"]
    return loaded, weight_file


def save_state_dict(state_dict, output):
    output = Path(output)
    if output.suffix:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_file = output
    else:
        output.mkdir(parents=True, exist_ok=True)
        output_file = output / "model.safetensors"

    if output_file.suffix == ".safetensors":
        save_file(state_dict, str(output_file), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_file)

    return output_file


def verify_fusion(original_state_dict, fused_state_dict):
    original_keys = set(original_state_dict.keys())
    fused_keys = set(fused_state_dict.keys())
    removed_keys = sorted(original_keys - fused_keys)
    added_keys = sorted(fused_keys - original_keys)
    remaining_layer_rotations = sorted(k for k in fused_keys if ".H2" in k or ".H3" in k)

    return {
        "removed_keys": removed_keys,
        "added_keys": added_keys,
        "remaining_layer_rotations": remaining_layer_rotations,
        "kept_global_H1": "bert.H1" in fused_keys,
    }


def copy_sidecar_files(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_dir() or output_path.suffix:
        return

    for source in input_path.iterdir():
        if source.name in {"model.safetensors", "pytorch_model.bin"} or source.is_dir():
            continue
        target = output_path / source.name
        if not target.exists():
            shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description="Fuse per-layer H2/H3 rotations for spikingbert_rot_inf.py.")
    parser.add_argument("--input", default="snn_rot_new3_110M/step_550000")
    parser.add_argument("--output", default="snn_rot_new3_110M_fused/step_550000")
    parser.add_argument("--config", default="tokenizer_files")
    args = parser.parse_args()

    config = BertConfig.from_pretrained(args.config, local_files_only=True)
    state_dict, source_weight_file = load_state_dict(args.input)
    fused_state_dict = fuse_rotation_matrices_in_state_dict(state_dict, config)
    output_file = save_state_dict(fused_state_dict, args.output)
    copy_sidecar_files(args.input, args.output)

    report = verify_fusion(state_dict, fused_state_dict)
    report.update(
        {
            "source_weight_file": str(source_weight_file),
            "output_weight_file": str(output_file),
        }
    )

    report_file = Path(args.output)
    if report_file.suffix:
        report_file = report_file.parent / "fusion_report.json"
    else:
        report_file = report_file / "fusion_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved fused weights to {output_file}")
    print(f"Saved fusion report to {report_file}")
    print(f"Removed {len(report['removed_keys'])} keys")
    print(f"Kept bert.H1: {report['kept_global_H1']}")
    print(f"Remaining per-layer H2/H3 keys: {len(report['remaining_layer_rotations'])}")


if __name__ == "__main__":
    main()

