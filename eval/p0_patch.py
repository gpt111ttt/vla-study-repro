"""P0 hooks: robot-init (qpos) and camera perturbations for LIBERO under lerobot-eval.

Why this funnel
---------------
lerobot's ``LiberoEnv.reset()`` does:

    self._env.reset()
    raw_obs = self._env.set_init_state(self._init_states[init_state_id])   # qpos set here
    for _ in range(10): self._env.step(dummy)                              # settle

``set_init_state`` -> ``regenerate_obs_from_state`` is therefore the single
place where the episode's start state is written, *before* the settle steps but
*after* robosuite's own reset. Patching it means the policy's very first
observation already reflects the perturbed start pose, and the settle steps do
not drag the arm back (controller goals are resynced).

Camera perturbation is applied further downstream, in ``LiberoEnv._format_raw_obs``,
i.e. on every observation (reset *and* every step), as an affine warp of each
returned image. That is an image-level stand-in for moving the sim camera, and it
is applied identically across arms, so it stays a fair paired comparison.

Env vars
--------
P0_QPOS_DELTA  float, metres/radians added to each arm joint (0.0 = no-op)
P0_QPOS_MODE   "fixed" (all joints +delta) | "random" (uniform(-d, d) per joint)
P0_CAM_ZOOM    float, magnification, 1.0 = no-op, 1.5 = crop-in 1.5x
P0_CAM_ROT     float, degrees of in-plane rotation (0.0 = no-op)
P0_CAM_SHIFT   float, fraction of image width/height to translate (0.0 = no-op)
P0_CAM_KEYS    comma-separated substrings of image keys to perturb
               (default "agentview,image,base_0_rgb")

Probe result it is built on (2026-11-02, vstc):
  robot.init_qpos assignment  -> no effect (reset overwrites)
  robot_model.initial_joint_positions -> attribute absent on MountedPanda
  sim.data.qpos[_ref_joint_pos_indexes] += d  -> moves the arm exactly (max|dq|=0.3000)
"""
from __future__ import annotations

import os
import sys

import numpy as np

# cuBLAS needs this set before CUDA is initialised, otherwise its workspace size (and
# therefore the reduction order) can differ between runs.
if os.environ.get("P0_CUBLAS_WS", "1") == "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

DELTA = float(os.environ.get("P0_QPOS_DELTA", "0.0"))
MODE = os.environ.get("P0_QPOS_MODE", "fixed")
CAM_ZOOM = float(os.environ.get("P0_CAM_ZOOM", "1.0"))
CAM_ROT = float(os.environ.get("P0_CAM_ROT", "0.0"))
CAM_SHIFT = float(os.environ.get("P0_CAM_SHIFT", "0.0"))
CAM_KEYS = tuple(k for k in os.environ.get(
    "P0_CAM_KEYS", "pixels,agentview,image,base_0_rgb").split(",") if k)
_CALL = {"n": 0, "cam": 0}

CAM_ACTIVE = CAM_ZOOM != 1.0 or CAM_ROT != 0.0 or CAM_SHIFT != 0.0
# The training dataset (base_0_rgb) is 180 deg rotated w.r.t. the simulator's raw
# agentview_image (measured: corr(raw, dataset) = 0.916 under rot180, -0.24 under
# identity). The policy expects the dataset convention, so rotate the observation.
IMG_ROT180 = os.environ.get("P0_IMG_ROT180", "0") == "1"
# Blind the policy: replace every image with a constant. If success does not move,
# the policy is not using the pixels at all.
IMG_ZERO = os.environ.get("P0_IMG_ZERO", "")  # "" off, else a 0-255 level
# The wrist camera renders 3 of 196608 pixels 1 grey level apart between two identical
# runs, and that is enough to flip ~13% of episodes. Round every image to a multiple of
# P0_QUANT grey levels so a +-1 render difference lands on the same value.
QUANT = int(os.environ.get("P0_QUANT", "1"))
DUMP = os.environ.get("P0_DUMP_DIR", "")
# Deterministic rollouts: flow matching draws fresh noise for every chunk. Without a
# fixed seed two identical runs disagree on ~8/30 episodes, which makes per-episode
# pairing meaningless. Seed per (task_id, init_state_id) and per chunk index.
SEED_ON = os.environ.get("P0_SEED", "1") == "1"
# P0_SEED_OFFSET makes repeated runs of the same arm draw different (but still paired)
# noise, which is how you measure run-to-run spread without changing the init states.
SEED_OFFSET = int(os.environ.get("P0_SEED_OFFSET", "0"))
# Pin episode k to init_states[k]. LiberoEnv.step() calls self.reset() on termination,
# and reset() advances init_state_id, so without this the state schedule advances by
# (1 + number of successes) per episode: two runs that differ in outcome end up
# evaluating different initial conditions, and the drift compounds. Observed on the
# 50k checkpoint: 30 episodes burned 55 resets on task 0 but only 37 on task 3.
PIN_INIT_STATES = os.environ.get("P0_PIN_INIT_STATES", "1") == "1"
_S = {"base": 0, "k": 0, "ep": 0, "in_step": False}


def _install_seed() -> None:
    import torch

    from lerobot.envs.libero import LiberoEnv

    # cuDNN autotuning picks different kernels between runs; switch it off so a fixed
    # seed actually reproduces the same arithmetic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # SDPA silently picks flash / mem-efficient / math depending on free memory, and on
    # LiCO that made two identically seeded runs disagree on 2 of 30 episodes. Pin the
    # backend so kernel choice stops depending on the allocator state.
    if os.environ.get("P0_DET_SDPA", "1") == "1":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            print("[p0] sdpa pinned: flash+math", file=sys.stderr)
        except Exception as exc:  # older torch builds
            print(f"[p0] sdpa pin skipped: {exc}", file=sys.stderr)
    # warn_only: prints every op without a deterministic implementation instead of
    # aborting the run. NOTE: this variant triggered a CUDA index_select assert on the
    # pi05 checkpoint, so it stays opt-in until the offending op is identified.
    if os.environ.get("P0_DET_ALGO", "0") == "1":
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("[p0] deterministic algorithms: on (warn_only)", file=sys.stderr)
        except Exception as exc:
            print(f"[p0] deterministic mode skipped: {exc}", file=sys.stderr)

    orig_reset = LiberoEnv.reset
    orig_step = LiberoEnv.step

    def step(self, *a, **kw):  # noqa: ANN001
        _S["in_step"] = True
        try:
            return orig_step(self, *a, **kw)
        finally:
            _S["in_step"] = False

    LiberoEnv.step = step

    def reset(self, *a, **kw):  # noqa: ANN001
        tid = int(getattr(self, "task_id", 0))
        auto = _S["in_step"]  # True when called from step() on termination
        saved = int(getattr(self, "init_state_id", 0))
        if not auto and PIN_INIT_STATES:
            # episode index, not the reset count, decides the initial condition
            ep_i = int(getattr(self, "_p0_ep_i", 0))
            self._p0_ep_i = ep_i + 1
            self.init_state_id = ep_i
        out = orig_reset(self, *a, **kw)
        if auto:
            # the observation from this reset is discarded by the caller; do not let it
            # consume an initial condition
            if PIN_INIT_STATES:
                self.init_state_id = saved
        else:
            eid = int(getattr(self, "_p0_ep_i", 1)) - 1
            _S["base"] = (SEED_OFFSET + tid * 1_000_003 + eid * 7919) % (2**31 - 7)
            _S["k"] = 0
            _S["ep"] += 1
            print(f"[p0] seed ep{_S['ep']} task={tid} init_state={eid} base={_S['base']}",
                  file=sys.stderr, flush=True)
        return out

    LiberoEnv.reset = reset

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch

    target = None
    for cls in (PI05Pytorch, PI05Policy):
        if "sample_actions" in cls.__dict__:
            target = cls
            break
    if target is None:
        print("[p0] seed: no sample_actions found, seeds NOT applied", file=sys.stderr, flush=True)
        return
    orig_sample = target.sample_actions
    # P0_DUMP_DIR: save the first action chunk of every episode as .npy so two runs can be
    # compared at the model output, separating model-side noise from environment noise.
    dump_dir = os.environ.get("P0_DUMP_DIR", "")

    def sample(self, *a, **kw):  # noqa: ANN001
        first = _S["k"] == 0
        torch.manual_seed(_S["base"] + _S["k"] * 104729)
        _S["k"] += 1
        out = orig_sample(self, *a, **kw)
        if dump_dir and first:
            try:
                import numpy as _np
                arr = out.detach().float().cpu().numpy() if hasattr(out, "detach") else _np.asarray(out)
                os.makedirs(dump_dir, exist_ok=True)
                _np.save(os.path.join(dump_dir, f"ep{_S['ep']:03d}_chunk0.npy"), arr)
            except Exception as exc:
                print(f"[p0] dump failed: {exc}", file=sys.stderr, flush=True)
        return out

    target.sample_actions = sample
    print(f"[p0] seeded sampling on {target.__name__}.sample_actions (episode-keyed)",
          file=sys.stderr, flush=True)
    if dump_dir:
        print(f"[p0] dumping first action chunk to {dump_dir}", file=sys.stderr, flush=True)


def _zero_images(obs, level: float):
    if not isinstance(obs, dict):
        return obs
    for k, v in list(obs.items()):
        if isinstance(v, dict):
            obs[k] = _zero_images(v, level)
        elif isinstance(v, np.ndarray) and v.ndim == 3:
            obs[k] = np.full_like(v, level)
    return obs


def _quantize_images(obs, step: int):
    """Round every image channel to a multiple of `step` grey levels (0-255 scale).

    Works for HWC uint8/float arrays and for CHW torch tensors. Values are rescaled to
    0-255, quantised, then mapped back to the original dtype and range.
    """
    if step <= 1 or not isinstance(obs, dict):
        return obs

    def one(v):
        try:
            import torch  # local import: stays optional
        except Exception:
            torch = None
        if torch is not None and isinstance(v, torch.Tensor):
            if v.ndim not in (3, 4):
                return v
            f = v.float()
            scale = 255.0 if f.max() <= 1.5 else 1.0
            q = torch.round(f * scale)
            q = torch.round(q / step) * step
            q = (q / scale).clamp(0.0, 255.0 / scale) if scale != 1.0 else q.clamp(0.0, 255.0)
            return q.to(v.dtype)
        if isinstance(v, np.ndarray) and v.ndim in (3, 4):
            f = v.astype(np.float32)
            scale = 255.0 if f.max() <= 1.5 else 1.0
            q = np.round(f * scale)
            q = np.round(q / step) * step
            q = np.clip(q / scale, 0.0, 255.0 / scale) if scale != 1.0 else np.clip(q, 0.0, 255.0)
            return q.astype(v.dtype)
        return v

    for k, v in list(obs.items()):
        if isinstance(v, dict):
            obs[k] = _quantize_images(v, step)
        else:
            obs[k] = one(v)
    return obs


def _rot180_images(obs):
    if not isinstance(obs, dict):
        return obs
    for k, v in list(obs.items()):
        if isinstance(v, dict):
            obs[k] = _rot180_images(v)
        elif isinstance(v, np.ndarray) and v.ndim == 3:
            obs[k] = np.ascontiguousarray(v[::-1, ::-1])
    return obs


def _maybe_dump(raw_obs, out) -> None:
    """Save the first observation, raw and formatted, for image-convention checks."""
    import json
    import os as _os
    import sys as _sys

    import cv2

    if not DUMP or _CALL.get("dumped"):
        return
    _CALL["dumped"] = True
    _os.makedirs(DUMP, exist_ok=True)
    info = {}
    if isinstance(raw_obs, dict):
        for k, v in raw_obs.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                cv2.imwrite(_os.path.join(DUMP, f"raw_{k}.png"), v[..., ::-1])
                info[f"raw_{k}"] = list(v.shape)
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, np.ndarray) and v2.ndim == 3:
                        arr = v2 if v2.dtype == np.uint8 else (v2 * 255).clip(0, 255).astype(np.uint8)
                        cv2.imwrite(_os.path.join(DUMP, f"fmt_{k2}.png"), arr[..., ::-1])
                        info[f"fmt_{k2}"] = list(v2.shape)
    (_os.path.join(DUMP, "info.json"))
    with open(_os.path.join(DUMP, "info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"[p0] dumped obs to {DUMP}: {info}", file=_sys.stderr, flush=True)


def _warp(img: np.ndarray) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), CAM_ROT, CAM_ZOOM)
    m[0, 2] += CAM_SHIFT * w
    m[1, 2] += CAM_SHIFT * h
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    if img.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    return out


def _perturb_images(obs):
    """Apply the affine warp to every image-looking value in a dict."""
    import torch

    if not isinstance(obs, dict):
        return obs
    if not _CALL.get("keys_printed"):
        print(f"[p0] obs keys={list(obs.keys())}", file=sys.stderr, flush=True)
        _CALL["keys_printed"] = True
    for k, v in list(obs.items()):
        if isinstance(v, dict):  # e.g. key "pixels" -> {"image": ..., "image2": ...}
            obs[k] = _perturb_images(v)
            continue
        if not any(s in k for s in CAM_KEYS):
            continue
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().numpy()
            squeeze = False
            while t.ndim > 3:
                t = t[0]
                squeeze = True
            arr = t if t.ndim == 3 else None
            if arr is None:
                continue
            if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
                out = _warp(arr)
                out = torch.from_numpy(np.ascontiguousarray(np.transpose(out, (2, 0, 1)))).to(v.dtype)
            else:
                out = torch.from_numpy(np.ascontiguousarray(_warp(arr))).to(v.dtype)
            obs[k] = out.unsqueeze(0) if squeeze else out
        elif isinstance(v, np.ndarray) and v.ndim == 3:
            w = _warp(v)
            if os.environ.get("P0_CAM_DEBUG"):
                print(f"[p0] warped {k} np diff="
                      f"{float(np.abs(w.astype(np.float32) - v.astype(np.float32)).mean()):.2f}",
                      file=sys.stderr, flush=True)
            obs[k] = w
        _CALL["cam"] += 1
    return obs


def install() -> None:
    from libero.libero.envs.env_wrapper import ControlEnv

    if getattr(ControlEnv, "_p0_patched", False):
        print("[p0] already installed", file=sys.stderr, flush=True)
        return

    orig = ControlEnv.regenerate_obs_from_state

    def patched(self, mujoco_state):
        obs = orig(self, mujoco_state)
        if DELTA == 0.0:
            return obs
        try:
            robot = self.env.robots[0]
            idx = np.asarray(robot._ref_joint_pos_indexes).ravel()
            if MODE == "random":
                rng = np.random.default_rng(_CALL["n"])
                d = rng.uniform(-DELTA, DELTA, size=idx.size)
            else:
                d = np.full(idx.size, DELTA)
            self.env.sim.data.qpos[idx] += d
            self.env.sim.forward()
            robot.controller.update(force=True)
            self._update_observables(force=True)
            obs = self.env._get_observations()
            _CALL["n"] += 1
            print(
                f"[p0] ep{_CALL['n']} qpos+= {np.round(d, 3).tolist()} max|dq|={float(np.abs(d).max()):.3f}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[p0] perturbation FAILED: {exc!r}", file=sys.stderr, flush=True)
        return obs

    ControlEnv.regenerate_obs_from_state = patched
    ControlEnv._p0_patched = True
    print(f"[p0] installed mode={MODE} delta={DELTA}", file=sys.stderr, flush=True)

    if SEED_ON:
        _install_seed()

    if CAM_ACTIVE or DUMP or IMG_ROT180 or QUANT > 1:
        from lerobot.envs.libero import LiberoEnv

        orig_fmt = LiberoEnv._format_raw_obs

        def patched_fmt(self, raw_obs):
            out = orig_fmt(self, raw_obs)
            if DUMP:
                _maybe_dump(raw_obs, out)
            if IMG_ROT180:
                out = _rot180_images(out)
            if IMG_ZERO:
                out = _zero_images(out, float(IMG_ZERO))
            if QUANT > 1:
                out = _quantize_images(out, QUANT)
            return _perturb_images(out) if CAM_ACTIVE else out

        LiberoEnv._format_raw_obs = patched_fmt
        LiberoEnv._p0_cam_patched = True
        print(f"[p0] camera zoom={CAM_ZOOM} rot={CAM_ROT} shift={CAM_SHIFT} "
              f"keys={CAM_KEYS} quant={QUANT}", file=sys.stderr, flush=True)