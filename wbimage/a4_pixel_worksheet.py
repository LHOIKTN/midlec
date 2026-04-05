"""
원본 PNG를 A4 한 장에 들어가도록 축소·양자화한 뒤,
학생용 표에는 이진 문자열(00, 01, …)만 넣고 범례로 색칠 규칙을 안내하는 PDF를 만듭니다.

칸 크기(mm)로 ‘픽셀 개수’와 ‘인쇄 가독성’을 맞춥니다. 값을 줄이면 그리드가 촘촘해져 그림이 더 잘 보이고,
키우면 칸·글자가 커져 색칠하기 쉽습니다. PNG 알파(투명)가 있으면 전경의 바깥쪽 한 겹(누끼 외곽) 칸은
표·CSV에서 레벨 0(4단계면 이진 00)으로 맞춥니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from quantize_3level import (
    binary_for_level,
    level_from_thresholds,
    luminance,
    parse_thresholds,
    rgb_for_level,
)


def composite_rgba(img: Image.Image, bg: str) -> Image.Image:
    w, h = img.size
    if bg == "white":
        base = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    else:
        base = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    return Image.alpha_composite(base, img).convert("RGB")


def build_level_grid(
    rgb: Image.Image, levels: int, thresholds: list[float]
) -> list[list[int]]:
    w, h = rgb.size
    px = rgb.load()
    grid: list[list[int]] = []
    for y in range(h):
        row: list[int] = []
        for x in range(w):
            r, g, b = px[x, y]
            row.append(level_from_thresholds(luminance(r, g, b), thresholds))
        grid.append(row)
    return grid


def fit_grid_size(
    src_w: int,
    src_h: int,
    content_w_mm: float,
    content_h_mm: float,
    min_cell_mm: float,
) -> tuple[int, int]:
    """비율 유지하며 한 변의 칸 크기가 min_cell_mm 이상이 되도록 최대 그리드 크기."""
    max_cols = max(1, int(content_w_mm / min_cell_mm))
    max_rows = max(1, int(content_h_mm / min_cell_mm))
    scale = min(max_cols / src_w, max_rows / src_h)
    cols = max(1, int(src_w * scale))
    rows = max(1, int(src_h * scale))
    return cols, rows


def binary_legend_text(levels: int) -> tuple[str, str]:
    """학생용: 표 안은 이진수만, 범례는 이진수→색 설명."""
    title = "범례 — 칸에 적힌 이진수에 맞춰 색칠하세요"
    if levels == 4:
        body = (
            "00 → 검정(가장 어둡게)   ·   01 → 어두운 회색   ·   "
            "10 → 밝은 회색   ·   11 → 흰색에 가깝게(가장 밝게)"
        )
    else:
        body = (
            "00 → 검정(가장 어둡게)   ·   01 → 중간 회색   ·   "
            "10 → 흰색에 가깝게(가장 밝게)"
        )
    return title, body


# draw_page 의 add_axes 와 동일해야 함 (칸 크기·글자 계산용)
_FIG_AX = (0.06, 0.12, 0.88, 0.78)


def print_cell_font_and_linewidth(
    fig: plt.Figure, cols: int, rows: int, line_width_floor: float
) -> tuple[float, float]:
    """
    A4 figure 안에서 실제 한 칸이 차지하는 크기(포인트)로 글자(pt)·선 두께를 맞춤.
    """
    left, bottom, w_frac, h_frac = _FIG_AX
    fw_in, fh_in = fig.get_size_inches()
    cell_w_pt = (fw_in * w_frac / max(cols, 1)) * 72.0
    cell_h_pt = (fh_in * h_frac / max(rows, 1)) * 72.0
    cell_pt = min(cell_w_pt, cell_h_pt)
    # 두 글자(00~11)가 칸 안에 들어가도록; 인쇄에서 읽을 수 있는 하한
    fontsize = max(6.0, min(16.0, cell_pt * 0.42))
    auto_lw = max(0.5, min(1.6, cell_pt / 13.0))
    linewidth = max(line_width_floor, auto_lw)
    return fontsize, linewidth


def build_fg_mask_rgba(
    rgba: Image.Image, cols: int, rows: int, alpha_cutoff: int = 128
) -> list[list[bool]]:
    """원본 알파 기준 전경(누끼 안). 합성 배경색과 무관."""
    small = rgba.resize((cols, rows), Image.Resampling.NEAREST)
    px = small.load()
    w, h = small.size
    return [[px[x, y][3] >= alpha_cutoff for x in range(w)] for y in range(h)]


def apply_rgba_outline_as_level_zero(
    grid: list[list[int]], fg: list[list[bool]]
) -> None:
    """전경이면서 4방 이웃 중 투명(비전경)이나 바깥과 맞닿은 칸 → 레벨 0 (4단계면 00)."""
    rows = len(fg)
    if rows == 0:
        return
    cols = len(fg[0])
    for y in range(rows):
        for x in range(cols):
            if not fg[y][x]:
                continue
            if (
                x == 0
                or not fg[y][x - 1]
                or x == cols - 1
                or not fg[y][x + 1]
                or y == 0
                or not fg[y - 1][x]
                or y == rows - 1
                or not fg[y + 1][x]
            ):
                grid[y][x] = 0


def draw_uniform_black_grid(
    ax: plt.Axes, cols: int, rows: int, linewidth: float
) -> None:
    """모든 칸 경계가 동일한 검은 실선이 되도록 격자만 별도로 그림."""
    if cols <= 0 or rows <= 0:
        return
    kw: dict = {
        "color": "#000000",
        "linewidth": linewidth,
        "solid_capstyle": "butt",
        "solid_joinstyle": "miter",
        "zorder": 2,
        "clip_on": True,
    }
    for xi in range(cols + 1):
        ax.plot([xi, xi], [0, rows], **kw)
    for yi in range(rows + 1):
        ax.plot([0, cols], [yi, yi], **kw)


def setup_korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Malgun Gothic",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "DejaVu Sans",
    ]


def draw_page(
    fig: plt.Figure,
    grid: list[list[int]],
    *,
    levels: int,
    title: str,
    student_mode: bool,
    line_width: float,
) -> None:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    fig.clf()
    ax = fig.add_axes(list(_FIG_AX))
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(title, fontsize=14, y=0.97)

    fontsize, lw = print_cell_font_and_linewidth(fig, cols, rows, line_width)

    for y in range(rows):
        for x in range(cols):
            lev = grid[y][x]
            if student_mode:
                face = (1, 1, 1)
            else:
                rr, gg, bb = rgb_for_level(lev, levels)
                face = (rr / 255, gg / 255, bb / 255)
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="none",
                    linewidth=0,
                    zorder=1,
                )
            )
    draw_uniform_black_grid(ax, cols, rows, lw)

    if student_mode:
        for y in range(rows):
            for x in range(cols):
                lev = grid[y][x]
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    binary_for_level(lev, levels),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="black",
                    fontweight="semibold",
                    zorder=5,
                )

    # 범례는 격자 바로 아래쪽(figure 좌표 y가 클수록 위)에 둠
    legend_body_y = 0.112
    if student_mode:
        leg_title, leg_body = binary_legend_text(levels)
        fig.text(0.5, legend_body_y + 0.034, leg_title, ha="center", fontsize=10)
        fig.text(0.5, legend_body_y, leg_body, ha="center", fontsize=8.2)
    else:
        fig.text(
            0.5,
            legend_body_y + 0.03,
            "정답 페이지 (교사용) — 이진수 대응 밝기",
            ha="center",
            fontsize=10,
        )
        parts = [
            f"{binary_for_level(lev, levels)} → 단계 {lev}" for lev in range(levels)
        ]
        fig.text(
            0.5,
            legend_body_y,
            "   ·   ".join(parts),
            ha="center",
            fontsize=8.2,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="A4 픽셀 격자 색칠 활동지 PDF")
    p.add_argument("-i", "--input", type=Path, default=Path("001.png"))
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("001_worksheet_a4.pdf"),
        help="학생용 PDF (이진수 표만)",
    )
    p.add_argument(
        "-k",
        "--key-output",
        type=Path,
        default=None,
        help="정답 PDF (밝기). 생략 시 학생용 파일명에 _key (예: …_a4_key.pdf)",
    )
    p.add_argument("--levels", type=int, choices=(3, 4), default=4)
    p.add_argument("--thresholds", type=str, default=None)
    p.add_argument("--bg", choices=("white", "black"), default="white")
    p.add_argument(
        "--min-cell-mm",
        type=float,
        default=6.0,
        help="한 칸의 목표 최소 크기(mm). 작을수록 픽셀(칸) 수↑·그림 선명, 클수록 칸·글자↑·그림 뭉개짐(기본 6)",
    )
    p.add_argument(
        "--margin-x-mm",
        type=float,
        default=12.0,
        help="좌우 여백(mm)",
    )
    p.add_argument(
        "--title-top-mm",
        type=float,
        default=18.0,
        help="상단 제목 영역(mm)",
    )
    p.add_argument(
        "--legend-bottom-mm",
        type=float,
        default=28.0,
        help="하단 범례(mm)",
    )
    p.add_argument(
        "--line-width",
        type=float,
        default=0.5,
        help="격자 선 최소 두께; 실제 출력은 칸 크기에 맞춰 더 두껍게 자동 보정",
    )
    p.add_argument(
        "--student-only",
        action="store_true",
        help="학생용 페이지만 저장",
    )
    p.add_argument(
        "--key-only",
        action="store_true",
        help="정답(명도) 페이지만 저장",
    )
    p.add_argument(
        "--csv-grid",
        type=Path,
        default=None,
        help="축소된 격자를 CSV로 저장 (기본: 출력 PDF와 같은 이름 .csv)",
    )
    p.add_argument(
        "--no-outline-00",
        action="store_true",
        help="알파 기준 누끼 외곽 칸을 레벨 0(00)으로 강제하지 않음",
    )
    args = p.parse_args()

    setup_korean_font()

    if args.student_only and args.key_only:
        raise SystemExit("--student-only 와 --key-only 는 동시에 쓸 수 없습니다.")

    img = Image.open(args.input).convert("RGBA")
    src_w, src_h = img.size
    rgb_full = composite_rgba(img, args.bg)

    content_w_mm = 210.0 - 2 * args.margin_x_mm
    content_h_mm = 297.0 - args.title_top_mm - args.legend_bottom_mm
    cols, rows = fit_grid_size(
        src_w, src_h, content_w_mm, content_h_mm, args.min_cell_mm
    )
    small = rgb_full.resize((cols, rows), Image.Resampling.NEAREST)

    thresholds = parse_thresholds(args.thresholds, args.levels)
    grid = build_level_grid(small, args.levels, thresholds)

    alpha_min, _alpha_max = img.split()[3].getextrema()
    if (not args.no_outline_00) and alpha_min < 250:
        fg_for_outline = build_fg_mask_rgba(img, cols, rows)
        apply_rgba_outline_as_level_zero(grid, fg_for_outline)

    csv_path = args.csv_grid
    if csv_path is None:
        csv_path = args.output.with_suffix(".csv")

    import csv as csv_mod

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv_writer = csv_mod.writer(f, quoting=csv_mod.QUOTE_MINIMAL)
        csv_writer.writerow([f"c{x}" for x in range(cols)])
        for row in grid:
            csv_writer.writerow([binary_for_level(v, args.levels) for v in row])

    a4_in = (210.0 / 25.4, 297.0 / 25.4)
    fig = plt.figure(figsize=a4_in)

    page_title = args.input.name

    key_path: Path | None = None
    if not args.student_only:
        key_path = (
            args.key_output
            if args.key_output is not None
            else args.output.with_name(args.output.stem + "_key" + args.output.suffix)
        )

    if not args.key_only:
        with PdfPages(args.output) as pdf:
            draw_page(
                fig,
                grid,
                levels=args.levels,
                title=page_title,
                student_mode=True,
                line_width=args.line_width,
            )
            pdf.savefig(fig, dpi=300)

    if not args.student_only and key_path is not None:
        with PdfPages(key_path) as pdf:
            draw_page(
                fig,
                grid,
                levels=args.levels,
                title=page_title,
                student_mode=False,
                line_width=args.line_width,
            )
            pdf.savefig(fig, dpi=300)

    plt.close(fig)

    cell_w_mm = content_w_mm / cols
    cell_h_mm = content_h_mm / rows
    if not args.key_only:
        print(f"학생용 PDF: {args.output.resolve()}")
    if not args.student_only and key_path is not None:
        print(f"정답 PDF: {key_path.resolve()}")
    print(f"격자 CSV: {csv_path.resolve()} (축소 후 {cols}×{rows}, 셀=2진 문자열)")
    print(
        f"칸 크기(약): {cell_w_mm:.2f}mm × {cell_h_mm:.2f}mm "
        f"(내용 영역 {content_w_mm:.0f}×{content_h_mm:.0f}mm)"
    )
    if cell_w_mm < 4.3 or cell_h_mm < 4.3:
        print(
            "※ 칸/글자가 매우 작을 수 있습니다. --min-cell-mm 을 6.5~8 정도로 키워 보세요."
        )


if __name__ == "__main__":
    main()
