"""
이미지를 명도 기준으로 N단계(3 또는 4)로 양자화하고,
PNG와 픽셀 데이터(txt 또는 CSV)로 저장합니다.
4단계일 때 각 레벨은 2비트 이진 문자열(00, 01, 10, 11)로 CSV에 기록합니다.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def luminance(r: int, g: int, b: int) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def level_from_thresholds(y: float, thresholds: list[float]) -> int:
    """thresholds 길이 = 단계 수 - 1, 오름차순. 반환 0 .. len(thresholds)."""
    for i, t in enumerate(thresholds):
        if y <= t:
            return i
    return len(thresholds)


def rgb_for_level(level: int, num_levels: int) -> tuple[int, int, int]:
    if num_levels <= 1:
        return (0, 0, 0)
    v = round(level * 255 / (num_levels - 1))
    return (v, v, v)


def binary_for_level(level: int, num_levels: int) -> str:
    """레벨을 이진 문자열로 (필요한 최소 비트 수, 최소 2비트)."""
    bits = max(2, (num_levels - 1).bit_length())
    return format(level, f"0{bits}b")


def parse_thresholds(raw: str | None, levels: int) -> list[float]:
    if raw is None or raw.strip() == "":
        if levels == 3:
            return [85.0, 170.0]
        if levels == 4:
            return [64.0, 128.0, 192.0]
        raise ValueError(f"기본 임계값은 levels 3 또는 4만 지원합니다. levels={levels}")
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != levels - 1:
        raise ValueError(
            f"임계값은 {levels - 1}개 필요합니다(쉼표 구분). 입력: {len(parts)}개"
        )
    for a, b in zip(parts, parts[1:]):
        if a > b:
            raise ValueError("임계값은 오름차순이어야 합니다.")
    return parts


def main() -> None:
    p = argparse.ArgumentParser(
        description="명도 기준 픽셀 양자화 (3/4단계), CSV/TXT 출력"
    )
    p.add_argument("-i", "--input", type=Path, default=Path("001.png"), help="입력 PNG")
    p.add_argument(
        "-o", "--output-image", type=Path, help="출력 PNG (기본: 001_Nlevel.png)"
    )
    p.add_argument(
        "-d",
        "--output-data",
        type=Path,
        help="픽셀 데이터 파일 (기본: 확장자에 따라 001_pixels.csv 또는 .txt)",
    )
    p.add_argument(
        "--levels",
        type=int,
        choices=(3, 4),
        default=4,
        help="양자화 단계 수 (기본: 4)",
    )
    p.add_argument(
        "--format",
        dest="data_format",
        choices=("csv", "txt"),
        default="csv",
        help="픽셀 데이터 형식 (기본: csv)",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="쉼표 구분 임계값. 3단계면 2개(예: 85,170), 4단계면 3개(예: 64,128,192). 생략 시 균등 분할",
    )
    p.add_argument(
        "--bg",
        choices=("white", "black"),
        default="white",
        help="투명 픽셀 합성 배경",
    )
    p.add_argument(
        "--excel-bom",
        action="store_true",
        help="CSV를 UTF-8 BOM으로 저장 (엑셀에서 한글/인코딩 호환)",
    )
    args = p.parse_args()

    levels = args.levels
    thresholds = parse_thresholds(args.thresholds, levels)

    if args.output_image is None:
        args.output_image = Path(f"001_{levels}level.png")
    if args.output_data is None:
        ext = "csv" if args.data_format == "csv" else "txt"
        args.output_data = Path(f"001_pixels.{ext}")

    img = Image.open(args.input).convert("RGBA")
    w, h = img.size

    if args.bg == "white":
        bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        base = Image.alpha_composite(bg, img)
    else:
        bg = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        base = Image.alpha_composite(bg, img)

    rgb = base.convert("RGB")
    pixels = rgb.load()
    out_img = Image.new("RGB", (w, h))
    out_load = out_img.load()

    rows_data: list[list[str]] = []

    for y in range(h):
        row: list[str] = []
        for x in range(w):
            r, g, b = pixels[x, y]
            y_luma = luminance(r, g, b)
            lev = level_from_thresholds(y_luma, thresholds)
            out_load[x, y] = rgb_for_level(lev, levels)
            if args.data_format == "csv":
                row.append(binary_for_level(lev, levels))
            else:
                row.append(str(lev))
        rows_data.append(row)

    out_img.save(args.output_image)

    if args.data_format == "csv":
        enc = "utf-8-sig" if args.excel_bom else "utf-8"
        with args.output_data.open("w", newline="", encoding=enc) as f:
            writer = csv.writer(
                f,
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            writer.writerow([f"c{x}" for x in range(w)])
            for row in rows_data:
                writer.writerow(row)
    else:
        lines = [f"{w} {h}", ""]
        lines.extend(" ".join(row) for row in rows_data)
        args.output_data.write_text("\n".join(lines), encoding="utf-8")

    bits = max(2, (levels - 1).bit_length())
    print(f"저장: {args.output_image.resolve()}")
    print(f"저장: {args.output_data.resolve()}")
    if args.data_format == "csv":
        print(
            f"CSV: 각 셀은 {bits}비트 이진 문자열 "
            f"(4단계 시 00=가장 어두움 … 11=가장 밝음)"
        )
    else:
        print(f"TXT: 0..{levels - 1} 정수")
    print(f"크기: {w}x{h}, 단계: {levels}, 임계값: {thresholds}, 배경: {args.bg}")


if __name__ == "__main__":
    main()
