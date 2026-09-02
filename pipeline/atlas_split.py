"""Atlas 切分器：按真实内容边界切分 AI 网格图集，避免把旁边单元格的内容切进来。

两类图集两条路径（分层回退，最坏情况=旧版等分盲切）：
  1. 实线分隔图集：通过"整列/整行颜色均匀"信号在预期边界附近定位分隔沟，
     全部内部边界检出 → 以沟中心切分（整条沟丢弃），外框同步剔除。
  2. 无缝图集（人物间只有背景空隙、无实线，常见于 is_segmentation 抠图图集）：
     旧版回退等分盲切，而 AI 摆放人物并不严格等分 → 切割线劈进旁边人物肢体。
     现改为内容感知切分：按内容掩码（alpha/背景色差）做行/列投影找内容带，
     在相邻内容带的空隙中心切分；每个行条带独立求列边界（容忍上下行人物错位）。
切分后透明图集按 alpha bbox 收紧贴住内容；白底图集做 trim_gray_border 修剪残留灰边。
"""
import os

import numpy as np
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

# 内容感知切分参数（无缝图集路径，在降采样坐标系下）
_CA_DOWNSAMPLE_W = 1600   # 检测用降采样宽度（px）
_CA_ALPHA_MIN = 8         # alpha ≥ 8 的像素算内容（透明图集；滤掉量化噪声）
_CA_BG_DIFF = 25          # 白底图集：与背景色任一通道偏差 > 25 算内容
_CA_PROJ_THR_FRAC = 0.002 # 投影内容阈值：≥ 轴长 × 0.2% 的像素才算有内容
_CA_BAND_MIN_FRAC = 0.01  # 内容带窄于轴长 1% 视为疑似分隔线/噪点
_CA_BAND_SLIM_RATIO = 0.3 # 且窄于最宽带 × 0.3 才丢弃（防误杀窄姿势带）
_CA_SMOOTH_K = 3          # 投影平滑窗口（px）


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
    """沿某一轴找分隔线/外框，返回 (bounds, found, notes)。

    bounds: 长度 n_cells+1 的内容边界列表（降采样坐标）。
    found: 长度 n_cells+1 的布尔列表，标记每条边界是否检出分隔沟。
    在每个预期边界（含外缘）±_SEARCH_RATIO 窗口内找"包含锚点"的最长连续均匀段：
      - 内部边界找到 → 边界取均匀段中心（整条沟对半丢弃）
      - 外缘找到 → 内容从框内侧起算（防病理过裁，钳制在半轴内）
      - 没找到 / 均匀段过宽（> _MAX_GUTTER_RATIO，是无缝图集的背景间隙而非分隔线）
        → 精确等分（无缝图集回退，由内容感知路径接管）
    """
    max_gutter = max(_MIN_GUTTER * 2, int(axis_len * _MAX_GUTTER_RATIO))
    bounds = [0] * (n_cells + 1)
    found = [False] * (n_cells + 1)
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
            found[i] = True
            notes.append(f"{axis_label}{i}: gutter [{seg_lo},{seg_hi}) -> {bounds[i]}")
        else:
            reason = "too wide (background gap)" if best_s >= 0 else "none"
            bounds[i] = expected
            notes.append(f"{axis_label}{i}: no gutter ({reason}) -> exact {expected}")
    return bounds, found, notes


def detect_grid_boundaries(img, cols, rows):
    """检测网格图集的分隔线/外框（实线图集路径）。

    返回 (x_bounds, y_bounds, found, notes)：x_bounds 长 cols+1，y_bounds 长 rows+1
    （原图坐标），found 为对应边界的检出标记（原图坐标映射前在降采样坐标判定），
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
    x_bounds, x_found, x_notes = _find_gutter("x", lambda k: _column_is_uniform(small, k), sw, cols)
    y_bounds, y_found, y_notes = _find_gutter("y", lambda k: _row_is_uniform(small, k), sh, rows)
    if scale != 1.0:
        x_bounds = [round(b * scale) for b in x_bounds]
        y_bounds = [round(b * scale) for b in y_bounds]
    return x_bounds, y_bounds, x_found + y_found, x_notes + y_notes


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


# ---------------------------------------------------------------------------
# 内容感知切分（无缝图集路径）
# ---------------------------------------------------------------------------


def _content_mask(small):
    """内容掩码（bool 2D）：透明图集按 alpha，白底图集按与四角中位色的偏差。"""
    if small.mode == "RGBA":
        return np.asarray(small.getchannel("A")) >= _CA_ALPHA_MIN
    rgb = np.asarray(small.convert("RGB")).astype(np.int16)
    h, w = rgb.shape[:2]
    corners = np.concatenate([
        rgb[:20, :20].reshape(-1, 3), rgb[:20, max(0, w - 20):].reshape(-1, 3),
        rgb[max(0, h - 20):, :20].reshape(-1, 3),
        rgb[max(0, h - 20):, max(0, w - 20):].reshape(-1, 3)])
    med = np.median(corners, axis=0)
    return np.abs(rgb - med).max(axis=2) > _CA_BG_DIFF


def _find_bands(proj):
    """1D 内容带检测：返回 [(start, end)))] 连续内容段列表。

    投影先平滑再按阈值二值化；窄带（< 轴长 1% 且 < 最宽带 30%）视为
    分隔线残留/噪点丢弃，防止实线把内容带切碎。
    """
    n = len(proj)
    k = min(_CA_SMOOTH_K, n)
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(proj.astype(np.float32), kernel, mode="same")
    content = smooth > max(2.0, n * _CA_PROJ_THR_FRAC)
    bands = []
    i = 0
    while i < n:
        if content[i]:
            j = i
            while j < n and content[j]:
                j += 1
            bands.append((i, j))
            i = j
        else:
            i += 1
    if not bands:
        return []
    max_w = max(e - s for s, e in bands)
    min_w = max(2, int(n * _CA_BAND_MIN_FRAC))
    return [(s, e) for s, e in bands
            if not ((e - s) < min_w and (e - s) < max_w * _CA_BAND_SLIM_RATIO)]


def _bands_to_bounds(bands, n_cells, axis_len):
    """内容带 → 单元格边界（降采样坐标）；带数不匹配 n_cells 时返回 None。

    bounds 长 n_cells+1：外缘 0/axis_len（格内收紧兜底），内部边界取相邻带空隙中点。
    """
    if len(bands) != n_cells:
        return None
    bounds = [0] * (n_cells + 1)
    for i in range(1, n_cells):
        bounds[i] = (bands[i - 1][1] + bands[i][0]) // 2
    bounds[n_cells] = axis_len
    return bounds


def detect_content_boundaries(img, cols, rows):
    """内容感知网格边界检测（无缝图集路径）。

    行方向：整图行投影找 rows 个内容带；列方向：**每个行条带独立**做列投影
    找 cols 个内容带（容忍上下行人物左右错位——等分盲切切进旁人的主因）。
    任一轴/条带带数不匹配 → 该轴/条带回退精确等分（与旧版行为一致）。

    返回 (x_bounds_per_row, y_bounds, notes)，均为原图坐标；
    x_bounds_per_row[r] 为第 r 行条带的列边界（长 cols+1）。
    """
    w, h = img.size
    small = _normalize(img)
    if w > _CA_DOWNSAMPLE_W:
        small = small.resize((_CA_DOWNSAMPLE_W,
                              max(1, round(h * _CA_DOWNSAMPLE_W / w))), Image.NEAREST)
    sw, sh = small.size
    mask = _content_mask(small)

    notes = []
    y_b = _bands_to_bounds(_find_bands(mask.sum(axis=1)), rows, sh)
    if y_b is None:
        y_b = [round(i * sh / rows) for i in range(rows + 1)]
        notes.append(f"y: content bands != {rows}, fallback exact")
    else:
        notes.append(f"y: content bands -> {y_b}")
    y_bounds = [round(b * h / sh) for b in y_b]

    x_per_row = []
    for r in range(rows):
        x_b = _bands_to_bounds(_find_bands(mask[y_b[r]:y_b[r + 1]].sum(axis=0)), cols, sw)
        if x_b is None:
            x_b = [round(i * sw / cols) for i in range(cols + 1)]
            notes.append(f"x(row{r}): content bands != {cols}, fallback exact")
        else:
            notes.append(f"x(row{r}): content bands -> {x_b}")
        x_per_row.append([round(b * w / sw) for b in x_b])
    return x_per_row, y_bounds, notes


def _is_transparent_atlas(img):
    """是否为带显著透明背景的抠图图集（决定 bbox 收紧 vs 灰边修剪）。"""
    if img.mode != "RGBA":
        return False
    a = np.asarray(img.getchannel("A"))
    return bool((a == 0).mean() > 0.05)


def _tighten_alpha_bbox(cell):
    """透明图集单元格：按内容 bbox 收紧裁剪（留 ~1% padding），贴住人物。

    输出尺寸允许各不相同：下游 stop_motion.normalize_pose 会把每张姿势
    独立缩放到统一画布，morph 光流也发生在归一化之后。
    """
    if cell.mode != "RGBA":
        return cell
    a = np.asarray(cell.getchannel("A"))
    ys, xs = np.where(a >= _CA_ALPHA_MIN)
    if len(xs) == 0:
        return cell
    w, h = cell.size
    pad = max(4, round(min(w, h) * 0.01))
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return cell
    return cell.crop((x0, y0, x1, y1))


def split_atlas(atlas_path, cols, rows, out_paths, log_prefix="[AtlasSplit]"):
    """按真实内容边界切分图集并保存各单元格，返回各格尺寸列表。

    out_paths 顺序为行优先：(0,0) (0,1) ... (1,0) ...，长度须等于 cols*rows。
    保留原图模式（RGBA 抠图图集不会丢失透明背景）。

    路径选择：实线分隔图集（全部内部边界检出分隔沟）走沟中心切分；
    无缝图集走内容感知切分（行条带独立列边界 + 空隙中心切分）；
    检测失败逐条回退等分，最坏情况与旧版等分盲切一致。
    """
    atlas = _normalize(Image.open(atlas_path))
    x_bounds, y_bounds, found, notes = detect_grid_boundaries(atlas, cols, rows)
    for note in notes:
        print(f"  {log_prefix} boundary {note}")

    gutters_found = all(found[1:cols]) and all(found[cols + 2:cols + 1 + rows])
    if gutters_found:
        print(f"  {log_prefix} path: gutter lines ({cols}x{rows})")
        cells = [(x_bounds[c], y_bounds[r], x_bounds[c + 1], y_bounds[r + 1])
                 for r in range(rows) for c in range(cols)]
    else:
        x_per_row, y_ca, ca_notes = detect_content_boundaries(atlas, cols, rows)
        for note in ca_notes:
            print(f"  {log_prefix} content {note}")
        cells = [(x_per_row[r][c], y_ca[r], x_per_row[r][c + 1], y_ca[r + 1])
                 for r in range(rows) for c in range(cols)]

    transparent = _is_transparent_atlas(atlas)
    sizes = []
    for idx, out_path in enumerate(out_paths):
        cell = atlas.crop(cells[idx])
        before = cell.size
        cell = _tighten_alpha_bbox(cell) if transparent else trim_gray_border(cell)
        cell.save(out_path)
        trimmed = (f" trim {before[0] - cell.size[0]}x{before[1] - cell.size[1]}"
                   if cell.size != before else "")
        print(f"  {log_prefix} Split: {os.path.basename(out_path)} "
              f"({cell.size[0]}x{cell.size[1]}{trimmed})")
        sizes.append(cell.size)
    return sizes
