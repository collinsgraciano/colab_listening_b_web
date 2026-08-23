"""Atlas 切分器：检测 AI 网格图集的分隔线/外框，按真实边界切分并修剪残边。

AI 生成的 "2x2/4x2 grid sheet" 常带 ~30-80px 的浅灰分隔线与外框，
按等分盲切会把分隔线劈进单元格（白边），且真实面板边界常偏离等分线。
本模块通过"整列/整行颜色均匀"信号在预期边界附近定位分隔线：
  - 检出 → 以分隔线中心为边界切分（整条沟丢弃），外框同步剔除
  - 未检出（无缝图集）→ 回退精确等分，行为与旧版盲切一致
切分后每格再做 trim_gray_border 兜底修剪残留灰边（上限 6%）。
"""
import os

from PIL import Image

# 检测参数（在降采样坐标系下）
_DOWNSAMPLE_W = 1600      # 检测用降采样宽度（px）
_SAMPLE_STEP = 3          # 沿均匀方向每 3px 采样 1 个像素
_COLOR_TOL = 18           # 像素与所在列/行中位色的偏差容差
_UNIFORM_RATIO = 0.90     # 列/行被判为"均匀"的像素占比阈值
_MIN_GUTTER = 3           # 均匀段至少 3px（降采样坐标）才算分隔线
_SEARCH_RATIO = 0.08      # 在预期边界 ±8% 范围内搜索分隔线
_MAX_GUTTER_RATIO = 0.04  # 分隔线宽度上限 4%；更宽的均匀段=无缝图集的背景间隙
# 修剪参数（原图坐标）
_TRIM_MAX_FRAC = 0.06     # 单边最多修剪该边长度的 6%
_TRIM_UNIFORM_RATIO = 0.92
_TRIM_COLOR_TOL = 20
_TRIM_MIN_BRIGHT = 140    # 只修中性偏亮的边（灰/白），不碰深色/透明内容


def _normalize(img):
    """模式归一化：调色板/灰度等特殊模式转 RGB(A)，保证 px[x,y] 返回颜色元组。"""
    if img.mode == "P":
        return img.convert("RGBA")
    if img.mode not in ("RGB", "RGBA"):
        return img.convert("RGB")
    return img


def _column_is_uniform(img, x):
    """列 x 是否"纵向均匀"：与中位色偏差 < _COLOR_TOL 的采样像素占比 ≥ 阈值。"""
    px = img.load()
    h = img.size[1]
    samples = [px[x, y][:3] for y in range(0, h, _SAMPLE_STEP)]
    return _uniform_ratio(samples) >= _UNIFORM_RATIO


def _row_is_uniform(img, y):
    """行 y 是否"横向均匀"。"""
    px = img.load()
    w = img.size[0]
    samples = [px[x, y][:3] for x in range(0, w, _SAMPLE_STEP)]
    return _uniform_ratio(samples) >= _UNIFORM_RATIO


def _uniform_ratio(samples):
    n = len(samples)
    if n == 0:
        return 0.0
    med = sorted(samples, key=lambda p: p[0] + p[1] + p[2])[n // 2]
    hit = sum(1 for p in samples
              if abs(p[0] - med[0]) < _COLOR_TOL
              and abs(p[1] - med[1]) < _COLOR_TOL
              and abs(p[2] - med[2]) < _COLOR_TOL)
    return hit / n


def _find_gutter(axis_label, is_uniform, axis_len, n_cells):
    """沿某一轴找分隔线/外框，返回 (bounds, notes)。

    bounds: 长度 n_cells+1 的内容边界列表（降采样坐标）。
    在每个预期边界（含外缘）±_SEARCH_RATIO 窗口内找"包含锚点"的最长连续均匀段：
      - 内部边界找到 → 边界取均匀段中心（整条沟对半丢弃）
      - 外缘找到 → 内容从框内侧起算（防病理过裁，钳制在半轴内）
      - 没找到 / 均匀段过宽（> _MAX_GUTTER_RATIO，是无缝图集的背景间隙而非分隔线）
        → 精确等分（无缝图集回退，行为与盲切一致）
    """
    max_gutter = max(_MIN_GUTTER * 2, int(axis_len * _MAX_GUTTER_RATIO))
    bounds = [0] * (n_cells + 1)
    notes = []
    positions = [round(i * axis_len / n_cells) for i in range(n_cells + 1)]
    for i, expected in enumerate(positions):
        window = max(_MIN_GUTTER * 2, int(axis_len * _SEARCH_RATIO))
        lo = max(0, expected - window)
        hi = min(axis_len, expected + window + 1)
        flags = [is_uniform(k) for k in range(lo, hi)]
        anchor = expected - lo
        # 找包含锚点（或贴住图缘）的最长连续均匀段
        best_s, best_e = -1, 0
        k = 0
        while k < len(flags):
            if flags[k]:
                j = k
                while j < len(flags) and flags[j]:
                    j += 1
                ok = (k <= anchor < j) or (i == 0 and k == 0) or (i == n_cells and j == len(flags))
                if ok and (j - k) > (best_e - best_s):
                    best_s, best_e = k, j
                k = j
            else:
                k += 1
        if (best_s >= 0
                and _MIN_GUTTER <= (best_e - best_s) <= max_gutter):
            seg_lo, seg_hi = lo + best_s, lo + best_e  # [seg_lo, seg_hi)
            if i == 0:
                bounds[i] = min(seg_hi, axis_len // 2)
            elif i == n_cells:
                bounds[i] = max(seg_lo, axis_len - axis_len // 2)
            else:
                bounds[i] = (seg_lo + seg_hi) // 2
            notes.append(f"{axis_label}{i}: gutter [{seg_lo},{seg_hi}) -> {bounds[i]}")
        else:
            reason = "too wide (background gap)" if best_s >= 0 else "none"
            bounds[i] = expected
            notes.append(f"{axis_label}{i}: no gutter ({reason}) -> exact {expected}")
    return bounds, notes


def detect_grid_boundaries(img, cols, rows):
    """检测网格图集的真实内容边界。

    返回 (x_bounds, y_bounds, notes)：x_bounds 长 cols+1，y_bounds 长 rows+1（原图坐标），
    单元格 (r, c) 的裁剪框为 (x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1])。
    内部降采样到宽 ≤ _DOWNSAMPLE_W 加速，结果按比例映射回原图。
    """
    w, h = img.size
    scale = 1.0
    small = _normalize(img)
    if w > _DOWNSAMPLE_W:
        scale = w / _DOWNSAMPLE_W
        small = small.resize((_DOWNSAMPLE_W, round(h * _DOWNSAMPLE_W / w)), Image.NEAREST)
    sw, sh = small.size
    x_bounds, x_notes = _find_gutter("x", lambda k: _column_is_uniform(small, k), sw, cols)
    y_bounds, y_notes = _find_gutter("y", lambda k: _row_is_uniform(small, k), sh, rows)
    if scale != 1.0:
        x_bounds = [round(b * scale) for b in x_bounds]
        y_bounds = [round(b * scale) for b in y_bounds]
    return x_bounds, y_bounds, x_notes + y_notes


def _trim_side(cell, side):
    """单边修剪：返回该边应裁掉的像素数（0 = 无需修剪）。"""
    px = cell.load()
    w, h = cell.size
    max_trim = int((w if side in ("left", "right") else h) * _TRIM_MAX_FRAC)

    def uniform_line(k):
        if side == "left":
            samples = [px[k, y][:3] for y in range(0, h, 3)]
        elif side == "right":
            samples = [px[w - 1 - k, y][:3] for y in range(0, h, 3)]
        elif side == "top":
            samples = [px[x, k][:3] for x in range(0, w, 3)]
        else:
            samples = [px[x, h - 1 - k][:3] for x in range(0, w, 3)]
        n = len(samples)
        if n == 0:
            return False
        med = sorted(samples, key=lambda p: p[0] + p[1] + p[2])[n // 2]
        r, g, b = med
        if r < _TRIM_MIN_BRIGHT or max(r, g, b) - min(r, g, b) > 30:  # 深色/彩色/透明不修
            return False
        hit = sum(1 for p in samples
                  if abs(p[0] - r) < _TRIM_COLOR_TOL
                  and abs(p[1] - g) < _TRIM_COLOR_TOL
                  and abs(p[2] - b) < _TRIM_COLOR_TOL)
        return hit / n >= _TRIM_UNIFORM_RATIO

    trimmed = 0
    for k in range(max_trim):
        if uniform_line(k):
            trimmed = k + 1
        else:
            break
    return trimmed


def trim_gray_border(cell):
    """修剪单元格四边的残留浅灰/白边（各边上限 6%），返回新 Image（无灰边时原样返回）。"""
    left = _trim_side(cell, "left")
    right = _trim_side(cell, "right")
    top = _trim_side(cell, "top")
    bottom = _trim_side(cell, "bottom")
    w, h = cell.size
    if left + right >= w or top + bottom >= h:  # 安全阀：避免裁空
        return cell
    if left or right or top or bottom:
        cell = cell.crop((left, top, w - right, h - bottom))
    return cell


def split_atlas(atlas_path, cols, rows, out_paths, log_prefix="[AtlasSplit]"):
    """按真实网格边界切分图集并保存各单元格，返回各格尺寸列表。

    out_paths 顺序为行优先：(0,0) (0,1) ... (1,0) ...，长度须等于 cols*rows。
    保留原图模式（RGBA 抠图图集不会丢失透明背景）。
    """
    atlas = _normalize(Image.open(atlas_path))
    x_bounds, y_bounds, notes = detect_grid_boundaries(atlas, cols, rows)
    for note in notes:
        print(f"  {log_prefix} boundary {note}")

    sizes = []
    for idx, out_path in enumerate(out_paths):
        r, c = divmod(idx, cols)
        cell = atlas.crop((x_bounds[c], y_bounds[r], x_bounds[c + 1], y_bounds[r + 1]))
        before = cell.size
        cell = trim_gray_border(cell)
        cell.save(out_path)
        trimmed = (f" trim {before[0] - cell.size[0]}x{before[1] - cell.size[1]}"
                   if cell.size != before else "")
        print(f"  {log_prefix} Split: {os.path.basename(out_path)} "
              f"({cell.size[0]}x{cell.size[1]}{trimmed})")
        sizes.append(cell.size)
    return sizes
