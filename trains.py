import json
import re
import sys


def format_num(n: float) -> str:
    """
    格式化数字：
    - 保留最多 2 位小数
    - 去掉多余的 0
    - 避免出现 -0
    """
    if abs(n) < 1e-9:
        n = 0.0
    s = f"{n:.2f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def convert_output(output_str: str) -> str:
    """
    从 A.json 的 output 中提取 [x, y]
    并转换成 B.json 风格的 output:
    [(x1,y1),(x2,y2),...]
    
    坐标变换规则：
    A: [x, y]
    B: (-y, x)
    """
    pattern = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    matches = re.findall(pattern, output_str)

    if not matches:
        return '[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]'

    converted_points = []
    for x_str, y_str in matches:
        x = float(x_str)
        y = float(y_str)

        bx = -y
        by = x

        converted_points.append(f"({format_num(bx)},{format_num(by)})")

    return "[" + ",".join(converted_points) + "]"


def process_item(item: dict) -> dict:
    """
    只修改 output 字段，其余字段保持不变
    """
    new_item = dict(item)
    if "output" in new_item and isinstance(new_item["output"], str):
        new_item["output"] = convert_output(new_item["output"])
    if "answer" in new_item and isinstance(new_item["answer"], str):
        new_item["answer"] = convert_output(new_item["answer"])
    return new_item


def main():
    input_path = "/cache/wx1427092/experiment/8/Qwen3-plan.json"
    # input_path = "/cache/wx1427092/experiment/8/val_5.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持：单个对象 / 对象数组
    if isinstance(data, list):
        new_data = [process_item(item) for item in data]
    elif isinstance(data, dict):
        new_data = process_item(data)
    else:
        raise ValueError("A.json 必须是 JSON 对象或对象数组")

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成，已输出到: {input_path}")


if __name__ == "__main__":
    main()