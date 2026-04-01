#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
import random

CAM_ORDER_1 = [
    "CAM_FRONT",
]

CAM_ORDER_3 = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
]

CAM_ORDER_6 = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

CAM_ORDER = CAM_ORDER_6

PROMPT_POOL = [
    "",
    "Use the multi-view images together with the historical motion states to infer a geometrically consistent future trajectory. The predicted waypoints should be smooth, physically plausible, and consistent with the recent motion trend, steering direction, and road geometry. ",

    "When predicting the future trajectory, prioritize safety, stability, smoothness, and conservative motion. Avoid aggressive turning or acceleration unless clearly supported by the scene and recent motion history. Prefer a physically feasible trajectory that maintains lane-following behavior and reduces risk in uncertain situations.",

    "Use a hierarchical planning process internally: first understand the surrounding scene and likely driving intent, then infer the short-term maneuver, and finally predict the future waypoint sequence. The final trajectory should be coherent with the visual scene, historical motion, and likely driving behavior.",
    
    "Use the recent trajectory, velocity, acceleration, and steering trend as strong motion cues. Predict a future trajectory that smoothly continues the current motion tendency unless the visual scene clearly indicates a change in maneuver. The predicted waypoints should remain temporally smooth and dynamically plausible.",
]

random.seed(42)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fmt_num(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    s = f"{v:.2f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    if "." not in s:
        s += ".0"
    return s


def ensure_xy_list(points: Any) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(points, list):
        return out
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                out.append([float(p[0]), float(p[1])])
            except Exception:
                continue
    return out


def right_front_to_forward_left(points: Sequence[Sequence[float]]) -> List[List[float]]:
    converted: List[List[float]] = []
    for p in points:
        if len(p) < 2:
            continue
        x_right = float(p[0])
        y_front = float(p[1])
        converted.append([y_front, -x_right])
    return converted


def get_history_points_forward_left(sample: Dict[str, Any]) -> List[List[float]]:
    if "history_traj_forward_left" in sample:
        return ensure_xy_list(sample.get("history_traj_forward_left"))
    if "history_traj_right_front" in sample:
        return right_front_to_forward_left(ensure_xy_list(sample.get("history_traj_right_front")))
    return []


def get_future_points_forward_left(sample: Dict[str, Any]) -> List[List[float]]:
    if "future_waypoints_forward_left" in sample:
        return ensure_xy_list(sample.get("future_waypoints_forward_left"))
    if "future_waypoints_right_front" in sample:
        return right_front_to_forward_left(ensure_xy_list(sample.get("future_waypoints_right_front")))
    return []


def get_state_value(state: Dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    for key in keys:
        if key in state and state[key] is not None:
            return state[key]
    return default


def get_accel_forward_left(state: Dict[str, Any]) -> List[float]:
    if isinstance(state.get("acceleration_ax_ay_forward_left"), (list, tuple)) and len(state["acceleration_ax_ay_forward_left"]) >= 2:
        return [float(state["acceleration_ax_ay_forward_left"][0]), float(state["acceleration_ax_ay_forward_left"][1])]
    acc_rf = state.get("acceleration_ax_ay_right_front")
    if isinstance(acc_rf, (list, tuple)) and len(acc_rf) >= 2:
        ax_right = float(acc_rf[0])
        ay_front = float(acc_rf[1])
        return [ay_front, -ax_right]
    return [0.0, 0.0]


def get_speed(state: Dict[str, Any], ego: Dict[str, Any]) -> float:
    for src in (state, ego):
        for key in ("speed",):
            if key in src and src[key] is not None:
                try:
                    return float(src[key])
                except Exception:
                    pass
    return 0.0


def get_steering(state: Dict[str, Any], ego: Dict[str, Any]) -> float:
    for src in (state, ego):
        for key in ("steering_angle", "steering_deg"):
            if key in src and src[key] is not None:
                try:
                    return float(src[key])
                except Exception:
                    pass
    return 0.0


def format_history_segment(history_points: List[List[float]], history_states: List[Dict[str, Any]], ego: Dict[str, Any]) -> str:
    count = len(history_points)

    segments = ""
    for idx, point in enumerate(history_points):
        steps_back = count - idx
        time_back = steps_back * 0.5
        state = history_states[idx] if idx < len(history_states) and isinstance(history_states[idx], dict) else {}
        acc = get_accel_forward_left(state)
        speed = get_speed(state, ego)
        steering = get_steering(state, ego)
        segments += (
            f"(t-{fmt_num(time_back)}s) [{fmt_num(point[0])}, {fmt_num(point[1])}], "
            f"Acceleration: X {fmt_num(acc[0])}, Y {fmt_num(acc[1])} m/s^2, "
            f"Velocity: {fmt_num(speed)} m/s, "
            f"Steering angle: {fmt_num(steering)} (positive: left turn, negative: right turn), "
        )
    # segments += "(t-0.0s) [0.0, 0.0]"
    # return segments

    curr_acc = get_accel_forward_left(ego)
    curr_speed = get_speed(ego, ego)
    curr_steering = get_steering(ego, ego)
    segments += (
        f"(t-0.0s) [0.0, 0.0], "
        f"Acceleration: X {fmt_num(curr_acc[0])}, Y {fmt_num(curr_acc[1])} m/s^2, "
        f"Velocity: {fmt_num(curr_speed)} m/s, "
        f"Steering angle: {fmt_num(curr_steering)} (positive: left turn, negative: right turn)"
    )
    return ", " + segments


def format_future_points(points: Sequence[Sequence[float]]) -> str:
    valid = ensure_xy_list(list(points))
    return ", ".join(f"[{fmt_num(p[0])}, {fmt_num(p[1])}]" for p in valid)


def reorder_images(images: Any) -> List[str]:
    if not isinstance(images, list):
        return []
    if len(images) != 6:
        return []
    image_map: Dict[str, str] = {}
    for path in images:
        if not isinstance(path, str):
            continue
        for cam in CAM_ORDER:
            if f"/{cam}/" in path or f"_{cam}_" in path or cam in path:
                image_map[cam] = path
                break
    if len(image_map) == 6:
        return [image_map[cam] for cam in CAM_ORDER]
    return [img for img in images if isinstance(img, str)]


def build_prompt(sample: Dict[str, Any]) -> str:
    ego = sample.get("ego_prompt_fields", {}) or {}
    history_points = get_history_points_forward_left(sample)
    history_states = sample.get("history_ego_states", [])
    if not isinstance(history_states, list):
        history_states = []

    history_seconds = len(history_points) * 0.5
    history_str = format_history_segment(history_points, history_states, ego)

    channel_order = ""
    image_placeholder = ""
    for channel in CAM_ORDER:
        channel_order += f"{channel} "
        image_placeholder += "\n<image>"
    return (
        "You are an autonomous driving agent."
        f"You have access to {len(CAM_ORDER)} surround-view camera image(s) of a vehicle in the following order:" 
        f"{channel_order}"
        f"{image_placeholder}. "
        "Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, "
        "given the vehicle's intent inferred from the images."
        f"Provided are the previous ego vehicle status recorded over the last {fmt_num(history_seconds)} seconds "
        "(at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. "
        "Positive x means forward direction while positive y means leftwards. "
        "The data is presented in the format [x, y]:"
        f"{history_str}\n"
        
        "Select the one driving strategy you think is most suitable for the current scenario from the four options below, and use it as a reference for waypoints prediction. "
        "1. Use the multi-view images to infer a future trajectory that is geometrically consistent with the visible road layout, lane structure, and drivable space. The predicted waypoints should remain spatially coherent with the scene geometry. "
        "2. Adopt a conservative planning policy that prioritizes stability and low-risk motion. Prefer controlled trajectory evolution over assertive maneuvering when multiple feasible futures exist. "
        "3. Use the recent trajectory, velocity, acceleration, and steering history as strong short-term motion priors. Predict a future trajectory that continues the current dynamic tendency in a physically feasible manner. "
        "4. Allocate more attention to lateral and rear surroundings, especially signals from adjacent lanes and vehicles approaching from the side or rear, as these cues may be important for short-horizon trajectory prediction. "

        "Predicted future movement details for the next 3 seconds "
        "(sampled at 0.5-second intervals), including BEV location in x and y directions "
        "(in meters). Positive x means forward direction while positive y means leftwards. "
        "The output is formatted as [x, y]: "
    )


def build_answer(sample: Dict[str, Any]) -> str:
    points = get_future_points_forward_left(sample)
    points_str = format_future_points(points)
    return (
        # "<PLANNING>Predicted future movement details for the next 3 seconds "
        # "(sampled at 0.5-second intervals), including BEV location in x and y directions "
        # "(in meters). Positive x means forward direction while positive y means leftwards. "
        # f"The output is formatted as [x, y]: {points_str}</PLANNING>"

        f"<PLANNING>{points_str}</PLANNING>"
    )


def build_record(sample: Dict[str, Any]) -> Dict[str, Any] | None:
    images = reorder_images(sample.get("image", []))
    token = sample.get("token") or sample.get("sample_token") or sample.get("id")

    if len(images) != 6 or not token:
        return None

    return {
        "image": images,
        # "image": [],
        "conversations": [
            {"from": "human", "value": build_prompt(sample)},
            {"from": "gpt", "value": build_answer(sample)},
        ],
        "sample_token": token,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="Path to input aligned json")
    parser.add_argument("--output-json", required=True, help="Path to output VQA json")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    data = load_json(args.input_json)
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("Unsupported input JSON structure.")

    out: List[Dict[str, Any]] = []
    skipped = 0
    for sample in samples:
        rec = build_record(sample)
        if rec is None:
            skipped += 1
            continue
        out.append(rec)
        if args.max_samples > 0 and len(out) >= args.max_samples:
            break

    dump_json(out, args.output_json)
    print(f"Wrote {len(out)} samples to {args.output_json}, skipped {skipped} invalid samples.")


if __name__ == "__main__":
    main()
