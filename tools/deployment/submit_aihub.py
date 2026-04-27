"""
Submit D-FINE pruned ONNX model to Qualcomm AI Hub for compilation + profiling.

Steps:
  1. Upload the FP32 ONNX
  2. Compile for a target Snapdragon device (INT8 via AI Hub's internal quantization)
  3. Profile inference latency + memory on-device

Usage:
    python tools/deployment/submit_aihub.py \
        --onnx output/pruning_recovery/best_recovery.onnx \
        --device "Samsung Galaxy S23"
"""

import argparse
import qai_hub


def main(args):
    if args.compiled_model_id:
        print(f"Skipping compile, using compiled model: {args.compiled_model_id}")
        compiled_model = qai_hub.get_model(args.compiled_model_id)
        device = qai_hub.Device(args.device)
        print("Submitting profile job...")
        profile_job = qai_hub.submit_profile_job(
            model=compiled_model,
            device=device,
        )
        print(f"Profile job: {profile_job}")
        profile_job.wait()
        profile_data = profile_job.download_profile()
        print("\n=== Profile Results ===")
        print(profile_data)
        return

    if args.model_id:
        print(f"Reusing uploaded model: {args.model_id}")
        model = qai_hub.get_model(args.model_id)
    else:
        print(f"Uploading {args.onnx} to Qualcomm AI Hub...")
        model = qai_hub.upload_model(args.onnx)
        print(f"Model uploaded: {model}")

    # Input specs: {name: (shape, dtype)} — dtypes from ONNX model inspection
    input_shapes = {
        "images": ((1, 3, 640, 640), "float32"),
        "orig_target_sizes": ((1, 2), "int64"),
    }

    device = qai_hub.Device(args.device)
    print(f"Target device: {args.device}")

    print("Submitting compile job (FP32 → INT8)...")
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_shapes,
        options="--quantize_full_type int8 --quantize_io --truncate_64bit_io",
    )
    print(f"Compile job: {compile_job}")
    print("Waiting for compilation...")
    compile_job.wait()

    status = compile_job.get_status()
    print(f"Compile status: {status}")

    if status.success:
        compiled_model = compile_job.get_target_model()
        print(f"Compiled model: {compiled_model}")

        print("Submitting profile job...")
        profile_job = qai_hub.submit_profile_job(
            model=compiled_model,
            device=device,
        )
        print(f"Profile job: {profile_job}")
        print("Waiting for profile results...")
        profile_job.wait()

        profile_data = profile_job.download_profile()
        print("\n=== Profile Results ===")
        print(profile_data)
    else:
        print(f"Compilation failed: {status.message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="output/pruning_recovery/best_recovery.onnx")
    parser.add_argument("--device", default="Samsung Galaxy S23",
                        help="AI Hub device name (run `qai-hub list-devices` to see options)")
    parser.add_argument("--model-id", default=None,
                        help="Reuse an already-uploaded model by ID (skip upload)")
    parser.add_argument("--compiled-model-id", default=None,
                        help="Skip compile, profile an already-compiled model by ID")
    args = parser.parse_args()
    main(args)
