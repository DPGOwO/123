#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

FUTURE_STEPS = 6
HISTORY_STEPS = 4


def load_pickle(path: str) -> Any:
    p = Path(path)

    with p.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            pass

    with p.open("rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except Exception:
            pass

    import torch

    try:
        return torch.load(str(p), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(p), map_location="cpu")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def maybe_round(v: float, ndigits: int = 4) -> float:
    return round(float(v), ndigits)


def as_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def extract_items(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("infos"), list):
        return [x for x in obj["infos"] if isinstance(x, dict)]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        out: List[Dict[str, Any]] = []
        for k, v in obj.items():
            if not isinstance(v, dict):
                continue
            item = dict(v)
            if "token" not in item and "sample_token" not in item:
                item["token"] = str(k)
            out.append(item)
        if out:
            return out
    raise ValueError("Unsupported pickle structure. Expected dict with infos, list[dict], or dict[token]=info.")


def get_token(info: Dict[str, Any]) -> Optional[str]:
    for key in ("token", "sample_token", "id"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def get_scene_token(info: Dict[str, Any]) -> Optional[str]:
    value = info.get("scene_token")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def index_by_token(items: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        tok = get_token(item)
        if tok:
            out[tok] = item
    return out


def rel_path(path_str: str) -> str:
    s = str(path_str).replace("\\", "/")
    for prefix in ("samples/", "sweeps/"):
        idx = s.find(prefix)
        if idx >= 0:
            return s[idx:]
    return s


def quat_wxyz_to_rotmat(q: Sequence[float]) -> np.ndarray:
    if len(q) != 4:
        raise ValueError(f"Quaternion must have 4 values, got {q}")
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = [float(v) for v in q]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def try_extract_pose(info: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pairs = [
        ("ego2global_translation", "ego2global_rotation"),
        ("ego_translation", "ego_rotation"),
        ("translation", "rotation"),
    ]
    for t_key, r_key in pairs:
        t = info.get(t_key)
        r = info.get(r_key)
        if t is not None and r is not None:
            t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
            r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
            if t_arr.shape[0] >= 3 and r_arr.shape[0] == 4:
                return t_arr[:3], r_arr

    ego_pose = info.get("ego_pose")
    if isinstance(ego_pose, dict):
        t = ego_pose.get("translation")
        r = ego_pose.get("rotation")
        if t is not None and r is not None:
            t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
            r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
            if t_arr.shape[0] >= 3 and r_arr.shape[0] == 4:
                return t_arr[:3], r_arr

    return None


def global_to_current_ego(current_t: np.ndarray, current_q: np.ndarray, global_xyz: np.ndarray) -> np.ndarray:
    rot = quat_wxyz_to_rotmat(current_q)
    return rot.T @ (global_xyz - current_t)


def rel_to_right_front(rel_xyz: Sequence[float]) -> List[float]:
    rel_x = float(rel_xyz[0])
    rel_y = float(rel_xyz[1])
    return [maybe_round(-rel_y), maybe_round(rel_x)]


def extract_images(info: Dict[str, Any], cam_order: Sequence[str]) -> List[str]:
    cams = info.get("cams")
    if isinstance(cams, dict):
        out: List[str] = []
        for cam in cam_order:
            cam_info = cams.get(cam)
            if not isinstance(cam_info, dict):
                return []
            path = (
                cam_info.get("data_path")
                or cam_info.get("img_path")
                or cam_info.get("filename")
                or cam_info.get("path")
            )
            if not path:
                return []
            out.append(rel_path(str(path)))
        return out

    img_filename = info.get("img_filename")
    if isinstance(img_filename, list):
        path_by_cam: Dict[str, str] = {}
        for raw_path in img_filename:
            path_str = rel_path(str(raw_path))
            for cam in cam_order:
                if f"/{cam}/" in path_str or f"__{cam}__" in path_str or cam in Path(path_str).name:
                    path_by_cam[cam] = path_str
                    break
        ordered = [path_by_cam[cam] for cam in cam_order if cam in path_by_cam]
        if len(ordered) == len(cam_order):
            return ordered

    return []


def to_points_xy(value: Any) -> List[List[float]]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        if arr.shape[0] % 2 != 0:
            return []
        arr = arr.reshape(-1, 2)
    elif arr.ndim >= 3:
        arr = arr.reshape(arr.shape[0], -1)
    out: List[List[float]] = []
    for row in arr:
        if len(row) >= 2:
            out.append([float(row[0]), float(row[1])])
    return out


def to_mask_list(value: Any, n: int) -> List[int]:
    if value is None:
        return [1] * n
    arr = np.asarray(value).reshape(-1).tolist()
    out = [1 if float(v) > 0.5 else 0 for v in arr[:n]]
    if len(out) < n:
        out.extend([1] * (n - len(out)))
    return out


def is_zero_point(point: Sequence[float], eps: float = 1e-6) -> bool:
    return len(point) >= 2 and abs(float(point[0])) <= eps and abs(float(point[1])) <= eps


def trim_current_point(raw: List[List[float]], mask: List[int]) -> Tuple[List[List[float]], List[int]]:
    if len(raw) >= FUTURE_STEPS + 1 and is_zero_point(raw[0]):
        return raw[1:], mask[1:]
    return raw, mask


def onehot_to_command(value: Any) -> Optional[str]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3:
        return None
    idx = int(np.argmax(arr[:3]))
    return ["Turn Right", "Turn Left", "Go Straight"][idx]


def extract_future_traj_right_front(info: Dict[str, Any]) -> Dict[str, Any]:
    raw = to_points_xy(info.get("gt_ego_fut_trajs"))
    if not raw:
        return {"future_right_front": [], "mask_first6": [], "is_valid": False}

    mask = to_mask_list(info.get("gt_ego_fut_masks"), len(raw))
    raw, mask = trim_current_point(raw, mask)

    valid_points: List[List[float]] = []
    valid_mask: List[int] = []
    for i, p in enumerate(raw[:FUTURE_STEPS]):
        if i >= len(mask):
            break
        if mask[i] != 1:
            break
        valid_points.append([maybe_round(float(p[0])), maybe_round(float(p[1]))])
        valid_mask.append(1)

    return {
        "future_right_front": valid_points,
        "mask_first6": valid_mask,
        "is_valid": len(valid_points) == FUTURE_STEPS,
    }


def build_history_from_prev_chain(
    info: Dict[str, Any],
    token_to_info: Dict[str, Dict[str, Any]],
    history_steps: int,
) -> List[List[float]]:
    current_pose = try_extract_pose(info)
    if current_pose is None:
        return []
    current_t, current_q = current_pose
    current_scene = get_scene_token(info)

    history_right_front: List[List[float]] = []
    cursor = info
    for _ in range(history_steps):
        prev_tok = str(cursor.get("prev", "")).strip()
        if not prev_tok:
            break
        prev_info = token_to_info.get(prev_tok)
        if prev_info is None:
            break
        prev_scene = get_scene_token(prev_info)
        if current_scene and prev_scene and prev_scene != current_scene:
            break
        prev_pose = try_extract_pose(prev_info)
        if prev_pose is None:
            break
        prev_t, _ = prev_pose
        rel = global_to_current_ego(current_t, current_q, prev_t)
        history_right_front.append(rel_to_right_front(rel))
        cursor = prev_info

    history_right_front.reverse()
    return history_right_front


def zero_ego_fields() -> Dict[str, Any]:
    return {
        "vx_vy_right_front": [0.0, 0.0],
        "ax_ay_right_front": [0.0, 0.0],
        "yaw_rate": 0.0,
        "speed": 0.0,
        "steering_deg": 0.0,
        "source": "zero",
    }


def convert_forward_left_to_right_front(x_forward: float, y_left: float) -> List[float]:
    return [maybe_round(-float(y_left)), maybe_round(float(x_forward))]


def compute_prompt_ego_fields(
    info: Dict[str, Any],
    token_to_info: Dict[str, Dict[str, Any]],
    wheelbase: float,
) -> Dict[str, Any]:
    lcf = info.get("gt_ego_lcf_feat")
    if lcf is None:
        lcf = info.get("lcf_feat")

    if lcf is not None:
        arr = np.asarray(lcf, dtype=np.float64).reshape(-1)
        if arr.shape[0] >= 9:
            vx_forward = float(arr[0])
            vy_left = float(arr[1])
            ax_forward = float(arr[2])
            ay_left = float(arr[3])
            yaw_rate = float(arr[4])
            speed = float(arr[7])
            steering = float(arr[8]) * 1.294
            if info['map_location'].startswith('singapore'):
                steering *= -1
            return {
                "vx_vy_right_front": convert_forward_left_to_right_front(vx_forward, vy_left),
                "ax_ay_right_front": convert_forward_left_to_right_front(ax_forward, ay_left),
                "yaw_rate": maybe_round(yaw_rate),
                "speed": maybe_round(speed),
                "steering_deg": maybe_round(steering),
                "source": "gt_ego_lcf_feat",
            }

    can_bus = info.get("can_bus")
    if can_bus is not None:
        arr = np.asarray(can_bus, dtype=np.float64).reshape(-1)
        if arr.shape[0] >= 15:
            vx_forward = float(arr[7])
            vy_left = float(arr[8])
            ax_forward = float(arr[10])
            ay_left = float(arr[11])
            yaw_rate = float(arr[13])
            speed = math.hypot(vx_forward, vy_left)
            steering = 0.0 if abs(speed) < 1e-6 else math.degrees(math.atan2(wheelbase * yaw_rate, speed))
            return {
                "vx_vy_right_front": convert_forward_left_to_right_front(vx_forward, vy_left),
                "ax_ay_right_front": convert_forward_left_to_right_front(ax_forward, ay_left),
                "yaw_rate": maybe_round(yaw_rate),
                "speed": maybe_round(speed),
                "steering_deg": maybe_round(steering),
                "source": "can_bus",
            }

    return zero_ego_fields()


def find_can_bus_payload(merged: Dict[str, Any]) -> Any:
    for key in ("can_bus", "gt_ego_lcf_feat", "lcf_feat"):
        if key in merged and merged[key] is not None:
            return merged[key]
    return None


def collect_history_ego_states(
    info: Dict[str, Any],
    token_to_info: Dict[str, Dict[str, Any]],
    wheelbase: float,
    history_steps: int,
) -> List[Dict[str, Any]]:
    history_states: List[Dict[str, Any]] = []
    current_scene = get_scene_token(info)
    cursor = info

    for _ in range(history_steps):
        prev_tok = str(cursor.get("prev", "")).strip()
        if not prev_tok:
            break

        prev_info = token_to_info.get(prev_tok)
        if prev_info is None:
            break

        prev_scene = get_scene_token(prev_info)
        if current_scene and prev_scene and prev_scene != current_scene:
            break

        prev_ego = compute_prompt_ego_fields(
            prev_info,
            token_to_info=token_to_info,
            wheelbase=wheelbase,
        )

        history_states.append(
            {
                "token": get_token(prev_info),
                "timestamp": prev_info.get("timestamp"),
                "velocity_vx_vy_right_front": prev_ego["vx_vy_right_front"],
                "acceleration_ax_ay_right_front": prev_ego["ax_ay_right_front"],
                "yaw_rate": prev_ego["yaw_rate"],
                "speed": prev_ego["speed"],
                "steering_deg": prev_ego["steering_deg"],
                "source": prev_ego.get("source", "zero"),
            }
        )

        cursor = prev_info

    history_states.reverse()
    return history_states


def build_sample(
    temporal_info: Dict[str, Any],
    cache_info: Optional[Dict[str, Any]],
    token_to_info: Dict[str, Dict[str, Any]],
    cam_order: Sequence[str],
    wheelbase: float,
    keep_raw_fields: bool,
) -> Optional[Dict[str, Any]]:
    merged = dict(temporal_info)
    if cache_info:
        merged.update(cache_info)

    token = get_token(merged)
    if not token:
        return None

    images = extract_images(merged, cam_order)
    if len(images) != len(cam_order):
        return None

    future = extract_future_traj_right_front(merged)
    if not future["is_valid"]:
        return None

    history = build_history_from_prev_chain(merged, token_to_info=token_to_info, history_steps=HISTORY_STEPS)
    history_ego_states = collect_history_ego_states(
        merged,
        token_to_info=token_to_info,
        wheelbase=wheelbase,
        history_steps=HISTORY_STEPS,
    )

    if len(history_ego_states) != len(history):
        min_len = min(len(history_ego_states), len(history))
        history = history[-min_len:] if min_len > 0 else []
        history_ego_states = history_ego_states[-min_len:] if min_len > 0 else []

    command = onehot_to_command(merged.get("gt_ego_fut_cmd"))
    ego_prompt = compute_prompt_ego_fields(merged, token_to_info=token_to_info, wheelbase=wheelbase)
    can_bus = find_can_bus_payload(merged)

    sample: Dict[str, Any] = {
        "token": token,
        "image": images,
        "navigation": command,
        "history_traj_right_front": history,
        "history_ego_states": history_ego_states,
        "future_waypoints_right_front": future["future_right_front"],
        "future_mask_first6": future["mask_first6"],
        "ego_prompt_fields": {
            "velocity_vx_vy_right_front": ego_prompt["vx_vy_right_front"],
            "acceleration_ax_ay_right_front": ego_prompt["ax_ay_right_front"],
            "yaw_rate": ego_prompt["yaw_rate"],
            "speed": ego_prompt["speed"],
            "steering_deg": ego_prompt["steering_deg"],
            "source": ego_prompt.get("source", "zero"),
        },
    }

    if can_bus is not None:
        sample["ego_prompt_fields"]["can_bus"] = as_list(can_bus)

    if keep_raw_fields:
        sample["raw"] = {
            "gt_ego_fut_cmd": as_list(merged.get("gt_ego_fut_cmd")),
            "gt_ego_fut_trajs": as_list(merged.get("gt_ego_fut_trajs")),
            "gt_ego_fut_masks": as_list(merged.get("gt_ego_fut_masks")),
            "gt_ego_fut_yaw": as_list(merged.get("gt_ego_fut_yaw")),
            "gt_ego_lcf_feat": as_list(merged.get("gt_ego_lcf_feat")),
            "gt_boxes": as_list(merged.get("gt_boxes")),
            "gt_names": as_list(merged.get("gt_names")),
            "gt_velocity": as_list(merged.get("gt_velocity")),
            "gt_agent_fut_trajs": as_list(merged.get("gt_agent_fut_trajs")),
            "gt_agent_fut_masks": as_list(merged.get("gt_agent_fut_masks")),
            "gt_agent_fut_yaw": as_list(merged.get("gt_agent_fut_yaw")),
            "gt_agent_lcf_feat": as_list(merged.get("gt_agent_lcf_feat")),
            "scene_token": merged.get("scene_token"),
            "timestamp": merged.get("timestamp"),
            "prev": merged.get("prev"),
            "next": merged.get("next"),
        }

    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert planning pickle data to right-front JSON with ego-state logic aligned to Impromptu-VLA (prefer gt_ego_lcf_feat, fallback can_bus). History is variable-length; future must have 6 valid waypoints.")
    parser.add_argument("--train-pkl", required=True, help="Path to nuscenes_infos_temporal_train.pkl or val.pkl")
    parser.add_argument("--cache-pkl", required=True, help="Path to cached_nuscenes_info.pkl")
    parser.add_argument("--out-json", required=True, help="Output JSON file")
    parser.add_argument(
        "--wheelbase",
        type=float,
        default=2.84,
        help="Wheelbase used for steering approximation.",
    )
    parser.add_argument(
        "--keep-raw-fields",
        action="store_true",
        help="Keep raw debugging fields in each sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 means export all samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    temporal_obj = load_pickle(args.train_pkl)
    cache_obj = load_pickle(args.cache_pkl)
    temporal_items = extract_items(temporal_obj)
    cache_items = extract_items(cache_obj)

    temporal_by_token = index_by_token(temporal_items)
    cache_by_token = index_by_token(cache_items)

    merged_for_lookup: Dict[str, Dict[str, Any]] = {}
    for tok, item in temporal_by_token.items():
        merged = dict(item)
        if tok in cache_by_token:
            merged.update(cache_by_token[tok])
        merged_for_lookup[tok] = merged

    samples: List[Dict[str, Any]] = []
    matched_cache = 0
    for tok, temporal_info in temporal_by_token.items():
        cache_info = cache_by_token.get(tok)
        if cache_info is not None:
            matched_cache += 1
        sample = build_sample(
            temporal_info=temporal_info,
            cache_info=cache_info,
            token_to_info=merged_for_lookup,
            cam_order=CAM_ORDER,
            wheelbase=float(args.wheelbase),
            keep_raw_fields=bool(args.keep_raw_fields),
        )
        if sample is None:
            continue
        samples.append(sample)
        if args.max_samples > 0 and len(samples) >= int(args.max_samples):
            break

    payload = {
        "metadata": {
            "format": "planning_json_right_front_v5_impromptu_ego_aligned",
            "num_samples": len(samples),
            "source_files": {
                "train_pkl": str(args.train_pkl),
                "cache_pkl": str(args.cache_pkl),
            },
            "cam_order": CAM_ORDER,
            "future_steps_max": FUTURE_STEPS,
            "history_steps_max": HISTORY_STEPS,
            "history_is_variable_length": True,
            "future_requires_exactly_6_valid_waypoints": True,
            "matched_cache_tokens": matched_cache,
            "keep_raw_fields": bool(args.keep_raw_fields),
        },
        "samples": samples,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()
