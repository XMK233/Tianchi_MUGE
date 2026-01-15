#!/usr/bin/env python3
"""
从指定日志文件中提取包含
"INFO:pipeline_template:[Branch] Entered at line xxx"（其中 xxx 为数字）
的行号，去重后输出到标准输出（或可选保存到文件）。

用法示例：
  python extract_branch_logs.py \
    /mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-IC/plan_1/sub_plan_2/util-useless.txt

可选参数：
  --output /path/to/save.txt   将结果写入指定文件
"""

import argparse
import sys
import re

PREFIX = "INFO:pipeline_template:[Branch] Entered at line"
PATTERN = re.compile(r"INFO:pipeline_template:\[Branch\] Entered at line\s+(\d+)")


def extract_numbers(input_path: str) -> list[int]:
    numbers = set()
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if PREFIX in line:
                    for m in PATTERN.finditer(line):
                        try:
                            numbers.add(int(m.group(1)))
                        except Exception:
                            # 非法数字忽略
                            pass
    except FileNotFoundError:
        print(f"[Error] File not found: {input_path}", file=sys.stderr)
    except Exception as e:
        print(f"[Error] Failed to read file: {e}", file=sys.stderr)
    return sorted(numbers)


def main():
    parser = argparse.ArgumentParser(description="Extract branch log lines from a text file")
    parser.add_argument(
        "input",
        nargs="?",
        default="/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-IC/plan_1/sub_plan_2/util-useless.txt",
        help="输入日志文件路径（默认为 util-useless.txt 的绝对路径）",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="将结果写入指定文件（默认打印到标准输出）",
    )
    args = parser.parse_args()

    numbers = extract_numbers(args.input)

    out_lines = [f"{n}\n" for n in numbers]

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
            print(f"[OK] Extracted {len(numbers)} unique line numbers to: {args.output}")
        except Exception as e:
            print(f"[Error] Failed to write output: {e}", file=sys.stderr)
            # 失败则退回到标准输出
            sys.stdout.writelines(out_lines)
    else:
        # 直接打印到标准输出
        sys.stdout.writelines(out_lines)


if __name__ == "__main__":
    main()
