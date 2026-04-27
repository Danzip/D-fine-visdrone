"""
Export a structurally-pruned D-FINE model to ONNX.

The pruned checkpoint stores ffn_dims=[k0, k1, k2] alongside the model weights.
Before loading the state_dict we resize each decoder FFN layer to match those dims,
then proceed identically to the standard export_onnx.py.

Usage:
    python tools/deployment/export_onnx_pruned.py \
        -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
        -r output/pruning_recovery/best_recovery.pth \
        --input-size 640 640 \
        --check --simplify
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn

from src.core import YAMLConfig


def get_decoder(model: nn.Module) -> nn.Module:
    """Navigate to the inner TransformerDecoder (same as recovery_train.py)."""
    return model.decoder.decoder


def apply_ffn_dims(model: nn.Module, ffn_dims: list):
    """Resize each decoder FFN layer to the pruned dimensions in-place."""
    for layer, new_dim in zip(get_decoder(model).layers, ffn_dims):
        old_linear1 = layer.linear1
        old_linear2 = layer.linear2
        # Rebuild with pruned hidden dim
        layer.linear1 = nn.Linear(old_linear1.in_features, new_dim,
                                  bias=old_linear1.bias is not None)
        layer.linear2 = nn.Linear(new_dim, old_linear2.out_features,
                                  bias=old_linear2.bias is not None)


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

    ffn_dims = checkpoint["ffn_dims"]
    ap = checkpoint.get("ap", "unknown")
    print(f"Pruned checkpoint: ffn_dims={ffn_dims}, AP={ap:.4f}")

    # Resize FFN layers before loading state_dict
    apply_ffn_dims(cfg.model, ffn_dims)

    state = checkpoint["model"]
    cfg.model.load_state_dict(state)
    print("Loaded pruned state_dict.")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    h, w = args.input_size
    data = torch.rand(1, 3, h, w)
    size = torch.tensor([[h, w]])
    _ = model(data, size)
    print(f"Test forward pass OK — input shape: {list(data.shape)}")

    dynamic_axes = {
        "images": {0: "N"},
        "orig_target_sizes": {0: "N"},
    }

    output_file = args.resume.replace(".pth", ".onnx")

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    print(f"Exported ONNX → {output_file}")

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")

    if args.simplify:
        import onnx
        import onnxsim
        input_shapes = {"images": list(data.shape), "orig_target_sizes": list(size.shape)}
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f"Simplified: {check}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, type=str)
    parser.add_argument("--resume", "-r", required=True, type=str)
    parser.add_argument("--input-size", nargs=2, type=int, default=[640, 640],
                        metavar=("H", "W"), help="Input H W (default: 640 640)")
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--simplify", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
