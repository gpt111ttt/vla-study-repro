"""LIBERO-Goal evaluation for OpenVLA under this paper's protocol.

Deliberately a separate file from the lerobot evaluation path. OpenVLA is not a
lerobot policy, and papering over the difference would hide exactly what the
experiment is about: one camera instead of two, no action chunking, discretised
actions, and a JPEG round-trip in preprocessing.

Two protocol hooks matter:

  --quant N        round every rendered pixel to a multiple of N before the model
                   sees it, at the same place the pi0.5 patch quantises
  --hash_steps K   record a hash of the raw and quantised image for the first K
                   steps of every episode, so "were the inputs bit-identical?"
                   can be answered from the log instead of assumed

Everything else follows openvla/openvla: experiments/robot/libero/ and
experiments/robot/openvla_utils.py (read 2026-11-09 via jsDelivr).
"""
import argparse
import hashlib
import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# LIBERO pickles its initial states with torch.save; torch >= 2.6 defaults weights_only=True
# and refuses to load them. Patch before any libero import touches a pickle.
_ORIG_TORCH_LOAD = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load

ACTION_DIM = 7
NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
EOS_SPACE = 29871  # what predict_action appends when the prompt does not already end with it
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--quant", type=int, default=1, help="quantise pixels to multiples of N; 1 = off")
    p.add_argument("--hash_steps", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--attn", default="eager")
    return p.parse_args()


def quantise(img, q):
    if q is None or q <= 1:
        return img
    return (np.round(img.astype(np.float32) / q) * q).clip(0, 255).astype(np.uint8)


def preprocess(obs, quant, size):
    """agent view -> 180 deg rotation -> JPEG round-trip -> lanczos resize.

    The rotation and the JPEG round-trip are both in the official pipeline. They are
    reproduced because they change which pixels reach the model, which is the whole
    subject of this study.
    """
    raw = obs["agentview_image"]
    hashed_raw = hashlib.sha1(raw.tobytes()).hexdigest()[:16]
    img = quantise(raw, quant)[::-1, ::-1]
    hashed_q = hashlib.sha1(img.tobytes()).hexdigest()[:16]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    buf.seek(0)
    pil = Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS)
    return pil, hashed_raw, hashed_q


def normalize_gripper_action(action, binarize=True):
    """Official form: [0,1] -> [-1,+1], then sign. The binarisation threshold is 0.5.

    The model emits the gripper in [0,1] with 0 = close. Mapping [0,1] -> [-1,+1] and
    taking the sign puts the threshold at 0.5; the subsequent inversion makes +1 = close.
    A sign applied to the raw value instead puts the threshold at 0, which turns every
    "close" request into "open". Pushing a drawer survives that; grasping does not.
    """
    action = np.array(action, dtype=np.float32).copy()
    action[-1] = 2.0 * action[-1] - 1.0
    if binarize:
        action[-1] = np.sign(action[-1])
    return action


def invert_gripper_action(action):
    action = np.array(action, dtype=np.float32).copy()
    action[-1] = action[-1] * -1.0
    return action


def load_policy(ckpt, attn):
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation=attn,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    stats_path = os.path.join(ckpt, "dataset_statistics.json")
    with open(stats_path) as f:
        model.norm_stats = json.load(f)
    return model, processor


def pick_unnorm_key(model, suite):
    if suite in model.norm_stats:
        return suite
    if suite + "_no_noops" in model.norm_stats:
        return suite + "_no_noops"
    raise SystemExit("no unnorm key for %s in %s" % (suite, list(model.norm_stats)))


def build_env(task, resolution=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # official value; deliberately not the paper's P0_SEED
    return env


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    max_steps = MAX_STEPS[args.suite]

    model, processor = load_policy(args.ckpt, args.attn)
    unnorm_key = pick_unnorm_key(model, args.suite)
    dtype = next(model.parameters()).dtype
    print("[eval] suite=%s task=%d episodes=%d quant=%d unnorm_key=%s dtype=%s"
          % (args.suite, args.task_id, args.n_episodes, args.quant, unnorm_key, dtype), flush=True)

    env = build_env(task)
    prompt = "In: What action should the robot take to %s?\nOut:" % task.language.lower()

    rows = []
    for ep in range(args.n_episodes):
        if ep >= len(init_states):
            print("[eval] only %d init states, stopping at %d" % (len(init_states), ep), flush=True)
            break
        env.reset()
        obs = env.set_init_state(init_states[ep])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        success = False
        hashes = []
        t0 = time.time()
        for step in range(max_steps):
            pil, h_raw, h_q = preprocess(obs, args.quant, args.resize)
            if step < args.hash_steps:
                hashes.append([step, h_raw, h_q])
            inputs = processor(prompt, pil).to("cuda", dtype=torch.bfloat16)
            # predict_action appends EOS_SPACE when the prompt does not already end with it,
            # but leaves attention_mask one short, which desynchronises the causal mask.
            if int(inputs["input_ids"][0, -1]) != EOS_SPACE:
                ids = torch.tensor([[EOS_SPACE]], dtype=inputs["input_ids"].dtype,
                                   device=inputs["input_ids"].device)
                mask = torch.ones((1, 1), dtype=inputs["attention_mask"].dtype,
                                  device=inputs["attention_mask"].device)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], dim=1)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], mask], dim=1)
            with torch.no_grad():
                action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            action = np.asarray(action, dtype=np.float32).reshape(-1)[:ACTION_DIM]
            action = np.clip(action, -1.0, 1.0)
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
        dt = time.time() - t0
        rows.append({"task": args.task_id, "episode": ep, "success": bool(success),
                     "steps": step + 1, "seconds": round(dt, 1), "hashes": hashes})
        print("[eval] ep=%d success=%s steps=%d %.1fs" % (ep, success, step + 1, dt), flush=True)

    env.close()
    n_ok = sum(r["success"] for r in rows)
    out = {"suite": args.suite, "task_id": args.task_id, "task": task.language,
           "quant": args.quant, "n": len(rows), "success": n_ok,
           "rate": (n_ok / len(rows)) if rows else None, "rows": rows}
    path = os.path.join(args.out, "eval_openvla_t%d_q%d.json" % (args.task_id, args.quant))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("[eval] %d/%d = %.3f -> %s" % (n_ok, len(rows), out["rate"] or 0, path), flush=True)


if __name__ == "__main__":
    main()