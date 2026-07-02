"""Attempt ONNX export for tencent/Hunyuan-0.5B-Instruct (browser / Transformers.js).

Optimum does not yet export hunyuan_v1_dense natively. This script tries torch dynamo export
for a minimal forward pass. On success, quantize and upload to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_fp32(model_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    print(f"Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Minimal single-step forward for export probe.
    messages = [{"role": "user", "content": "Hello"}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    print("Exporting ONNX (dynamo) …")
    export_path = onnx_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids,),
            str(export_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
            opset_version=17,
        )
    print(f"Wrote {export_path}")

    # Copy tokenizer + config for HF repo layout.
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        src = Path(model_id) if Path(model_id).is_dir() else None
        if src and (src / name).exists():
            shutil.copy(src / name, out_dir / name)
    print("Done. Quantize with onnxruntime and upload to Theicd/Hunyuan-0.5B-Instruct-ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tencent/Hunyuan-0.5B-Instruct")
    parser.add_argument("--out", default="tools/hunyuan-onnx-publish")
    args = parser.parse_args()
    export_fp32(args.model, Path(args.out))
