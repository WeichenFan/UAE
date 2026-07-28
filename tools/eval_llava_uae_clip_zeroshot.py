#!/usr/bin/env python3
"""
Zero-shot LLaVA evaluation with UAE-CLIP-L as the vision tower (no training).

Loads `llava-hf/llava-1.5-7b-hf` (or compatible) and optionally swaps its
CLIPVisionTransformer state dict with the vision branch of a UAE-CLIP-L
checkpoint. Defaults assume the UAE encoder is trained at LLaVA's native
336x336 (24x24 token grid), so no PE interpolation is needed.

Targets POPE-style yes/no questions (one JSON-lines file with fields
{"image", "text", "label"}). Reports accuracy / precision / recall / F1 /
yes-ratio. Runs single-GPU; eval is batch-1 for simplicity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llava-model", type=str, default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--uae-ckpt", type=str, required=True,
                   help="UAE-CLIP-L checkpoint (.pt). Used only when --mode=uae.")
    p.add_argument("--mode", choices=["baseline", "uae"], required=True)
    p.add_argument("--prefer-ema", action="store_true", default=True,
                   help="Prefer the 'ema' state dict in the UAE checkpoint.")
    p.add_argument("--no-prefer-ema", dest="prefer_ema", action="store_false")
    p.add_argument("--projector-ckpt", type=str, default="",
                   help="Optional path to a trained multi_modal_projector state dict. "
                        "If set (and --mode=uae), loaded after vision-tower swap so the "
                        "projector matches UAE's feature distribution.")
    p.add_argument("--questions", type=str, required=True,
                   help="JSON file (POPE format: list or jsonl) with image/text/label.")
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--encoder-grid-in", type=int, default=24,
                   help="UAE-CLIP-L token grid side; default 24 matches LLaVA-1.5 "
                        "native (336/14). Set to 16 only for legacy 224-trained ckpts.")
    p.add_argument("--encoder-grid-out", type=int, default=24,
                   help="LLaVA-1.5 token grid side (336/14 = 24).")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--do-sample", action="store_true", default=False,
                   help="Use sampling instead of greedy decoding (longer/diverse captions).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--prompt-style", choices=["pope_yesno", "describe"], default="pope_yesno",
                   help="pope_yesno: 'Is there X? Answer in single word/phrase.' "
                        "describe: 'Describe the image in detail.' (ignores POPE 'text', uses fixed prompt)")
    p.add_argument("--low-freq-keep", type=int, default=0,
                   help="If >0, apply channel-wise 2D DCT to projector inputs and keep "
                        "only the top-left NxN block (zero high-freq), then inverse-DCT. "
                        "Tests the Prism Hypothesis on whatever encoder is loaded.")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit to first N questions (0 = all).")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--predictions", type=str, default="",
                   help="Optional path to dump per-question predictions (jsonl).")
    p.add_argument("--print-every", type=int, default=100)
    return p.parse_args()


def load_uae_vision_state(ckpt_path: str, prefer_ema: bool = True) -> dict:
    """Pull just the CLIP vision_model weights out of a UAE checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and prefer_ema and "ema" in ckpt:
        sd = ckpt["ema"]
        src = "ema"
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
        src = "model"
    else:
        sd = ckpt
        src = "flat"
    print(f"[load] using '{src}' state from {ckpt_path}")

    prefix = "encoder.model.vision_model."
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    if not out:
        raise RuntimeError(
            f"No keys matched prefix '{prefix}' in checkpoint. "
            "Is this a CLIP-L UAE checkpoint?"
        )
    print(f"[load] extracted {len(out)} vision_model parameters")
    return out


def interpolate_position_embeddings(state_dict: dict, old_grid: int, new_grid: int) -> dict:
    """Bicubic-interp CLIP position embeddings from old_gridxold_grid to new_gridxnew_grid."""
    if old_grid == new_grid:
        return state_dict
    pe_key = "embeddings.position_embedding.weight"
    if pe_key not in state_dict:
        print(f"[pe-interp] {pe_key} not in state dict; skipping")
        return state_dict
    pe = state_dict[pe_key]
    expected_old = old_grid * old_grid + 1
    if pe.shape[0] != expected_old:
        print(f"[pe-interp] PE has {pe.shape[0]} positions, expected {expected_old}; skipping")
        return state_dict
    cls = pe[:1]
    grid = pe[1:]  # [old*old, dim]
    dim = grid.shape[1]
    grid = grid.reshape(old_grid, old_grid, dim).permute(2, 0, 1).unsqueeze(0).float()
    grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    grid = grid.squeeze(0).permute(1, 2, 0).reshape(new_grid * new_grid, dim).to(pe.dtype)
    state_dict[pe_key] = torch.cat([cls, grid], dim=0)
    print(f"[pe-interp] {old_grid}x{old_grid} -> {new_grid}x{new_grid} ({expected_old} -> {new_grid*new_grid+1} positions)")
    pid_key = "embeddings.position_ids"
    if pid_key in state_dict:
        state_dict[pid_key] = torch.arange(new_grid * new_grid + 1).unsqueeze(0)
    return state_dict


def load_pope_questions(path: str) -> list[dict]:
    """POPE files are JSON-lines with one object per line."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def parse_yes_no(answer: str) -> Optional[bool]:
    a = answer.strip().lower()
    # Strip leading punctuation
    while a and a[0] in '\'"`.,!?':
        a = a[1:]
    if a.startswith("yes"):
        return True
    if a.startswith("no"):
        return False
    if "yes" in a and "no" not in a:
        return True
    if "no" in a and "yes" not in a:
        return False
    return None


def main() -> int:
    args = parse_args()

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    device = "cuda:0"
    dtype = torch.float16

    print(f"[load] LLaVA: {args.llava_model}")
    processor = AutoProcessor.from_pretrained(args.llava_model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.llava_model, torch_dtype=dtype
    ).to(device)
    model.eval()

    if hasattr(model.config, "vision_feature_layer"):
        print(f"[info] vision_feature_layer={model.config.vision_feature_layer}, "
              f"vision_feature_select_strategy={getattr(model.config, 'vision_feature_select_strategy', 'default')}")

    # ---- optional: DCT low-pass filter on projector input ----
    if args.low_freq_keep > 0:
        cache: dict[tuple, torch.Tensor] = {}

        def _build_dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
            ni = torch.arange(n, device=device, dtype=dtype).view(1, n)
            m = torch.cos((torch.pi / n) * (ni + 0.5) * k)
            m[0] *= 1.0 / (n ** 0.5)
            if n > 1:
                m[1:] *= (2.0 / n) ** 0.5
            return m

        def get_mat(n: int, dev: torch.device, dt: torch.dtype) -> torch.Tensor:
            key = (n, dev.type, str(dt))
            if key not in cache:
                cache[key] = _build_dct_matrix(n, dev, dt)
            return cache[key]

        gs = args.encoder_grid_out
        kp = args.low_freq_keep

        def _filter(x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            if N != gs * gs:
                return x
            od = x.dtype
            xf = x.transpose(1, 2).reshape(B, C, gs, gs).float()
            m = get_mat(gs, xf.device, xf.dtype)
            xf = m @ xf
            xf = xf @ m.t()
            mask = torch.zeros_like(xf)
            mask[..., :kp, :kp] = 1.0
            xf = xf * mask
            xf = m.t() @ xf
            xf = xf @ m
            return xf.reshape(B, C, N).transpose(1, 2).contiguous().to(od)

        def _pre_hook(module, args):
            if not args:
                return None
            return (_filter(args[0]),) + args[1:]

        model.multi_modal_projector.register_forward_pre_hook(_pre_hook)
        print(f"[dct] low-pass hook on projector: keep {kp}x{kp}={kp**2}/{gs**2} lowest-freq tokens")

    if args.mode == "uae":
        vision_sd = load_uae_vision_state(args.uae_ckpt, prefer_ema=args.prefer_ema)
        vision_sd = interpolate_position_embeddings(
            vision_sd,
            old_grid=args.encoder_grid_in,
            new_grid=args.encoder_grid_out,
        )
        # LLaVA's tower is HF CLIPVisionModel; the inner module is .vision_model (CLIPVisionTransformer).
        target = model.vision_tower.vision_model
        result = target.load_state_dict(vision_sd, strict=False)
        miss, unexp = list(result.missing_keys), list(result.unexpected_keys)
        print(f"[load] missing={len(miss)} unexpected={len(unexp)}")
        if miss:
            print(f"[load]  first missing: {miss[:5]}")
        if unexp:
            print(f"[load]  first unexpected: {unexp[:5]}")
        # Convert any newly-loaded fp32 PEs back to fp16 to match the model dtype.
        target.to(device=device, dtype=dtype)

        if args.projector_ckpt:
            print(f"[load] projector ckpt: {args.projector_ckpt}")
            proj_sd = torch.load(args.projector_ckpt, map_location="cpu", weights_only=False)
            if isinstance(proj_sd, dict) and "state_dict" in proj_sd:
                proj_sd = proj_sd["state_dict"]
            # Strip a leading 'module.' if the checkpoint was saved from DDP.
            proj_sd = {k[len("module."):] if k.startswith("module.") else k: v
                       for k, v in proj_sd.items()}
            res = model.multi_modal_projector.load_state_dict(proj_sd, strict=False)
            print(f"[load]  projector missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
            model.multi_modal_projector.to(device=device, dtype=dtype)

    questions = load_pope_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"[eval] {len(questions)} questions from {args.questions}")

    correct = 0
    yes_pred = 0
    yes_label = 0
    yes_pred_for_yes_label = 0
    parse_failures = 0

    pred_log = None
    if args.predictions:
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        pred_log = open(args.predictions, "w")

    images_dir = Path(args.images_dir)

    for i, q in enumerate(questions):
        img_path = images_dir / q["image"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[skip] image load failed for {img_path}: {e}")
            continue

        question_text = q["text"].strip()
        if args.prompt_style == "describe":
            prompt = "USER: <image>\nDescribe the image in detail. ASSISTANT:"
        else:
            prompt = (
                f"USER: <image>\n{question_text}\n"
                "Answer the question using a single word or phrase. ASSISTANT:"
            )

        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
            if args.do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p
            out_ids = model.generate(**inputs, **gen_kwargs)
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred = parse_yes_no(answer)
        gt = q["label"].strip().lower() == "yes"
        if pred is None:
            parse_failures += 1
            pred = False  # conservative

        is_correct = pred == gt
        correct += int(is_correct)
        yes_pred += int(pred)
        yes_label += int(gt)
        if gt and pred:
            yes_pred_for_yes_label += 1

        if pred_log is not None:
            pred_log.write(json.dumps({
                "question_id": q.get("question_id", i),
                "image": q["image"],
                "text": question_text,
                "label": q["label"],
                "raw_answer": answer,
                "pred_yes": pred,
                "correct": is_correct,
            }) + "\n")

        if (i + 1) % args.print_every == 0:
            running_acc = correct / (i + 1)
            print(f"[{i+1}/{len(questions)}] acc={running_acc:.4f}  yes_ratio={yes_pred/(i+1):.3f}")

    if pred_log is not None:
        pred_log.close()

    n = len(questions)
    accuracy = correct / max(n, 1)
    yes_ratio = yes_pred / max(n, 1)
    recall = yes_pred_for_yes_label / max(yes_label, 1)
    precision = yes_pred_for_yes_label / max(yes_pred, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print()
    print(f"=== mode={args.mode} questions={Path(args.questions).name} ===")
    print(f"N: {n}")
    print(f"accuracy:  {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"yes_ratio: {yes_ratio:.4f}")
    print(f"parse_failures: {parse_failures}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "mode": args.mode,
                "llava_model": args.llava_model,
                "uae_ckpt": args.uae_ckpt if args.mode == "uae" else None,
                "projector_ckpt": args.projector_ckpt if args.mode == "uae" else None,
                "questions_file": args.questions,
                "n": n,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "yes_ratio": yes_ratio,
                "parse_failures": parse_failures,
            }, f, indent=2)
        print(f"[save] metrics -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
