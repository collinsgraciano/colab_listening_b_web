"""sleep 模式合成：组级块构建 + concat 三段式（防 AAC priming 漂移）。

每块 = 一张静态卡片（-loop 1）+ 该组音频链（朗读段 + anullsrc 静音气口，
filter_complex 内统一 aresample/立体声后 concat 单编码 aac）→ 200+ 个均匀
块（libx264/yuv420p/25fps/aac 44100 立体声）走 media_utils.concat_segments。
xfade_sec>0 时相邻块先交叉溶解合并为纯视频流（仅画面、音频仍走网格拼接），
失败自动回退硬切。
文字全部预渲染进卡片 → 无字幕烧录步骤；末尾 apply_final_loudnorm 原地归一。

native_4k=True 时卡片按 3840x2160 原生渲染、块直接编码 4K（文字像素级
清晰，跳过 Step 6 lanczos 放大）；False 输出与历史版本逐字节一致（720p）。
"""
import os
import shutil
import subprocess
from pathlib import Path

from media_utils import (TARGET_H, TARGET_W, apply_final_loudnorm,
                         concat_segments, get_duration, merge_blocks_xfade,
                         safe_filename)
from sleep.sleep_cards import (render_intro_card, render_outro_card,
                               render_pair_card)

BLOCK_TIMEOUT = 600


def _output_vf(out_w: int, out_h: int) -> str:
    """按输出分辨率构造 scale/pad（720p 时与 media_utils.VF_NORM 一致）。"""
    return (f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2")


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=BLOCK_TIMEOUT)


def _build_audio_chain(block_segs: list[dict], audio_paths: dict,
                       lead: float = 0.0) -> tuple[str, list[str]]:
    """块内音频 filter_complex：朗读文件 + 静音气口统一 44100 立体声 concat。

    返回 (filter_complex 字符串, ffmpeg 输入参数列表)。段音频按段类型查：
    pair → pair_paths[(pair, step)]；intro/outro → audio_paths["intro"/"outro"]；
    无音频段（gap）生成等长静音（anullsrc 须以 -f lavfi 输入，否则被当作
    文件名导致整块失败）。lead>0 时组首段（a_m）朗读前先补 lead 秒静音
    （卡片提前量：画面先出现、稍后出声；该段 timeline duration 已含 lead）。
    """
    inputs: list[str] = []
    chains: list[str] = []
    concat_refs: list[str] = []
    # 输入 0 = 卡片图（-loop 1，无音频流），音频输入索引从 1 起
    n_in = 1
    for seg in block_segs:
        path = ""
        seg_type = seg.get("type", "")
        if seg_type == "pair":
            key = str(seg.get("pair", 0)).zfill(4)
            path = (audio_paths.get("pair_paths", {}).get(key, {})
                    .get(seg.get("step", ""), ""))
        elif seg_type == "intro":
            path = audio_paths.get("intro", "")
        elif seg_type == "outro":
            path = audio_paths.get("outro", "")
        if path and os.path.exists(path):
            if seg_type == "pair" and seg.get("step") == "a_m" and lead > 0:
                inputs += ["-f", "lavfi", "-i",
                           f"anullsrc=r=44100:cl=stereo:d={lead:.3f}"]
                concat_refs.append(f"[{n_in}:a]")
                n_in += 1
            inputs += ["-i", path]
            chains.append(f"[{n_in}:a]aresample=44100,aformat=channel_layouts=stereo[a{n_in}]")
            concat_refs.append(f"[a{n_in}]")
        else:
            inputs += ["-f", "lavfi", "-i",
                       f"anullsrc=r=44100:cl=stereo:d={max(0.0, float(seg.get('duration', 0))):.3f}"]
            concat_refs.append(f"[{n_in}:a]")
        n_in += 1
    fg = ";".join(chains) + ";" + "".join(concat_refs) + f"concat=n={len(concat_refs)}:v=0:a=1[aout]"
    return fg, inputs


def _build_video_block(intro_video: str, block_segs: list[dict], out_path: str,
                       vf: str) -> None:
    """绑定片头视频时的 intro 块：整段转码统一规格（音画随片头自带）。"""
    block_dur = round(sum(float(seg.get("duration", 0.0)) for seg in block_segs), 3)
    cmd = ["ffmpeg", "-y", "-i", intro_video,
           "-vf", vf,
           "-t", f"{block_dur:.3f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
           out_path]
    r = _run_ffmpeg(cmd)
    if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError(f"FFmpeg intro video block failed: {(r.stderr or '')[-300:]}")


def _build_block(card_path: str, block_segs: list[dict], audio_paths: dict,
                 out_path: str, vf: str, lead: float = 0.0) -> None:
    """构建一个块 mp4（静态卡 + 音频链）。"""
    block_dur = round(sum(float(seg.get("duration", 0.0)) for seg in block_segs), 3)
    fg, inputs = _build_audio_chain(block_segs, audio_paths, lead=lead)
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", card_path]
    cmd += inputs  # 已含 "-i <file>" 与 "-f lavfi -i anullsrc=..." 完整参数片段
    cmd += ["-filter_complex", fg,
            "-map", "0:v:0", "-map", "[aout]",
            "-t", f"{block_dur:.3f}",
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            out_path]
    r = _run_ffmpeg(cmd)
    if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError(f"FFmpeg block failed ({Path(out_path).name}): "
                           f"{(r.stderr or '')[-300:]}")


def _ensure_cards(timeline: list[dict], script: dict, cards_dir: Path,
                  theme: dict, channel_name: str, badge_text: str,
                  outro_text: str, num_pairs: int,
                  w: int = TARGET_W, h: int = TARGET_H) -> dict:
    """文件级续传渲染卡片。返回 {card_key: path}，key: intro/0001../outro。

    4K 卡片文件名带 _4k 后缀，避免误复用历史 720p 卡片（跨尺寸续传错尺寸）。
    """
    cards_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_4k" if w != TARGET_W else ""
    cards: dict[str, str] = {}
    if any(seg.get("type") == "intro" for seg in timeline):
        intro_path = str(cards_dir / f"intro_card{suffix}.png")
        if not os.path.exists(intro_path):
            render_intro_card(theme, intro_path, channel_name, badge_text, w=w, h=h)
        cards["intro"] = intro_path
    dialogue = script.get("dialogue", [])
    rows_a, rows_b = dialogue[0::2], dialogue[1::2]
    for i in range(1, num_pairs + 1):
        path = str(cards_dir / f"card_{i:04d}{suffix}.png")
        if not os.path.exists(path):
            a = rows_a[i - 1] if i <= len(rows_a) else {}
            b = rows_b[i - 1] if i <= len(rows_b) else {}
            render_pair_card(a, b, i, theme, path, channel_name, badge_text,
                             w=w, h=h)
        cards[str(i).zfill(4)] = path
    outro_path = str(cards_dir / f"outro_card{suffix}.png")
    if not os.path.exists(outro_path):
        render_outro_card(theme, outro_path, outro_text, channel_name, badge_text,
                          w=w, h=h)
    cards["outro"] = outro_path
    return cards


def compose_sleep(work_dir: str, timeline: list[dict], script: dict,
                  audio_results: dict, cards_dir: str, theme: dict,
                  channel_name: str = "English with me", badge_text: str = "EN",
                  outro_text: str = "", num_pairs: int = 0,
                  intro_video: str = "", native_4k: bool = False,
                  card_lead: float = 0.0, xfade_sec: float = 0.0,
                  progress_cb=None, stop_check=None) -> str:
    """合成 sleep 成片。返回最终 mp4 路径（videos/{safe}.mp4）。

    native_4k=True：卡片原生 3840x2160 渲染 + 块编码 4K（成片即 4K，
    下游 Step 6 检测已 4K 自动硬链接跳过放大）。
    xfade_sec>0：相邻块边界（组间 + 片头/片尾衔接）画面交叉溶解过渡
    （0.2-2.0s；仅画面，音频不动；整片多 1-2 次视频重编码）。0=硬切。
    """
    out_w, out_h = (3840, 2160) if native_4k else (TARGET_W, TARGET_H)
    vf = _output_vf(out_w, out_h)
    work = Path(work_dir)
    vid_dir = work / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = work / "tmp_sleep_blocks"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    (tmp_dir / "blocks").mkdir(parents=True, exist_ok=True)

    def _cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    pairs_in_tl = [seg for seg in timeline if seg.get("type") == "pair"]
    max_pair = max((int(seg.get("pair", 0)) for seg in pairs_in_tl), default=0)
    _cb(2, f"Rendering cards (pairs={max_pair}, {out_w}x{out_h})...")
    cards = _ensure_cards(timeline, script, Path(cards_dir), theme,
                          channel_name, badge_text, outro_text, max_pair,
                          w=out_w, h=out_h)

    # --- 时间轴 → 块序列：intro | [5 pair + 5 gap]* | outro ---
    blocks: list[list[dict]] = []
    cur: list[dict] = []
    for seg in timeline:
        t = seg.get("type", "")
        if t in ("intro", "outro"):
            if cur:
                blocks.append(cur)
                cur = []
            blocks.append([seg])
        elif t == "pair":
            cur.append(seg)
        elif t == "gap":
            cur.append(seg)
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    block_paths: list[str] = []
    total = len(blocks)
    for bi, block_segs in enumerate(blocks):
        if stop_check and stop_check():
            raise RuntimeError("stopped")
        head = block_segs[0]
        t = head.get("type", "")
        out_path = str(tmp_dir / "blocks" / f"block_{bi:04d}.mp4")
        # 绑定片头视频：intro 块整段转码该片（音画随片头自带 BGM/播报）
        is_intro_video = (t == "intro" and intro_video
                          and os.path.exists(intro_video))
        if not is_intro_video:
            if t == "intro":
                card = cards["intro"]
            elif t == "outro":
                card = cards["outro"]
            else:
                card = cards[str(head.get("pair", 0)).zfill(4)]
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 1000):
            try:
                if is_intro_video:
                    _build_video_block(intro_video, block_segs, out_path, vf)
                else:
                    _build_block(card, block_segs, audio_results, out_path, vf,
                                 lead=card_lead)
            except RuntimeError as e:
                if str(e) == "stopped":
                    raise
                print(f"  [Sleep] Block {bi} failed ({e}), retry once...")
                if is_intro_video:
                    _build_video_block(intro_video, block_segs, out_path, vf)
                else:
                    _build_block(card, block_segs, audio_results, out_path, vf,
                                 lead=card_lead)
        block_paths.append(out_path)
        if bi % 10 == 0 or bi == total - 1:
            _cb(int(2 + bi / total * 78),
                f"Block {bi + 1}/{total} ({t}, {head.get('pair', '')})".strip())

    merged_video = None
    if xfade_sec > 0 and len(block_paths) >= 2:
        if native_4k:
            print("  [Sleep] 4K 原生 + 交叉溶解：整片重编码耗时较长，请耐心等待")
        _cb(80, f"Crossfading {len(block_paths)} blocks ({xfade_sec:.2f}s)...")
        merged_video = merge_blocks_xfade(
            block_paths, str(tmp_dir / "xfade_merged.mp4"), xfade_sec)
        if merged_video is None:
            print("  [Sleep] 交叉溶解合并失败 — 回退硬切拼接")
    _cb(82, "Concatenating blocks...")
    no_sub = str(vid_dir / "final_no_sub.mp4")
    concat_segments(block_paths, no_sub, tmp_dir=str(tmp_dir),
                    video_source=merged_video)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    _cb(92, "Final loudnorm...")
    apply_final_loudnorm(no_sub, str(vid_dir))

    yt_title = script.get("youtube_title") or script.get("title") or "sleep_video"
    final_path = vid_dir / f"{safe_filename(yt_title, 'sleep_video')}.mp4"
    os.replace(no_sub, final_path)
    dur = get_duration(str(final_path))
    _cb(100, f"Sleep video done: {final_path.name} ({dur / 60:.1f} min)")
    return str(final_path)
