"""sleep 模式合成：组级块构建 + concat 三段式（防 AAC priming 漂移）。

每块 = 一张静态卡片（-loop 1）+ 该组音频链（朗读段 + anullsrc 静音气口，
filter_complex 内统一 aresample/立体声后 concat 单编码 aac）→ 200+ 个均匀
块（libx264/yuv420p/25fps/aac 44100 立体声）走 media_utils.concat_segments。
文字全部预渲染进卡片 → 无字幕烧录步骤；末尾 apply_final_loudnorm 原地归一。
"""
import os
import shutil
import subprocess
from pathlib import Path

from media_utils import (VF_NORM, apply_final_loudnorm, concat_segments,
                         get_duration, safe_filename)
from sleep.sleep_cards import (render_intro_card, render_outro_card,
                               render_pair_card)

BLOCK_TIMEOUT = 600


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=BLOCK_TIMEOUT)


def _build_audio_chain(block_segs: list[dict], audio_paths: dict) -> tuple[str, list[str]]:
    """块内音频 filter_complex：朗读文件 + 静音气口统一 44100 立体声 concat。

    返回 (filter_complex 字符串, ffmpeg 输入参数列表)。段音频按段类型查：
    pair → pair_paths[(pair, step)]；intro/outro → audio_paths["intro"/"outro"]；
    无音频段（gap）生成等长静音（anullsrc 须以 -f lavfi 输入，否则被当作
    文件名导致整块失败）。
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


def _build_video_block(intro_video: str, block_segs: list[dict], out_path: str) -> None:
    """绑定片头视频时的 intro 块：整段转码统一规格（音画随片头自带）。"""
    block_dur = round(sum(float(seg.get("duration", 0.0)) for seg in block_segs), 3)
    cmd = ["ffmpeg", "-y", "-i", intro_video,
           "-vf", VF_NORM,
           "-t", f"{block_dur:.3f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
           out_path]
    r = _run_ffmpeg(cmd)
    if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError(f"FFmpeg intro video block failed: {(r.stderr or '')[-300:]}")


def _build_block(card_path: str, block_segs: list[dict], audio_paths: dict,
                 out_path: str) -> None:
    """构建一个块 mp4（静态卡 + 音频链）。"""
    block_dur = round(sum(float(seg.get("duration", 0.0)) for seg in block_segs), 3)
    fg, inputs = _build_audio_chain(block_segs, audio_paths)
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", card_path]
    cmd += inputs  # 已含 "-i <file>" 与 "-f lavfi -i anullsrc=..." 完整参数片段
    cmd += ["-filter_complex", fg,
            "-map", "0:v:0", "-map", "[aout]",
            "-t", f"{block_dur:.3f}",
            "-vf", VF_NORM,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            out_path]
    r = _run_ffmpeg(cmd)
    if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError(f"FFmpeg block failed ({Path(out_path).name}): "
                           f"{(r.stderr or '')[-300:]}")


def _ensure_cards(timeline: list[dict], script: dict, cards_dir: Path,
                  theme: dict, channel_name: str, badge_text: str,
                  outro_text: str, num_pairs: int) -> dict:
    """文件级续传渲染卡片。返回 {card_key: path}，key: intro/0001../outro。"""
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    intro_path = str(cards_dir / "intro_card.png")
    if not os.path.exists(intro_path):
        render_intro_card(theme, intro_path, channel_name, badge_text)
    cards["intro"] = intro_path
    dialogue = script.get("dialogue", [])
    rows_a, rows_b = dialogue[0::2], dialogue[1::2]
    for i in range(1, num_pairs + 1):
        path = str(cards_dir / f"card_{i:04d}.png")
        if not os.path.exists(path):
            a = rows_a[i - 1] if i <= len(rows_a) else {}
            b = rows_b[i - 1] if i <= len(rows_b) else {}
            render_pair_card(a, b, i, theme, path, channel_name, badge_text)
        cards[str(i).zfill(4)] = path
    outro_path = str(cards_dir / "outro_card.png")
    if not os.path.exists(outro_path):
        render_outro_card(theme, outro_path, outro_text, channel_name, badge_text)
    cards["outro"] = outro_path
    return cards


def compose_sleep(work_dir: str, timeline: list[dict], script: dict,
                  audio_results: dict, cards_dir: str, theme: dict,
                  channel_name: str = "English with me", badge_text: str = "EN",
                  outro_text: str = "", num_pairs: int = 0,
                  intro_video: str = "",
                  progress_cb=None, stop_check=None) -> str:
    """合成 sleep 成片。返回最终 mp4 路径（videos/{safe}.mp4）。"""
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
    _cb(2, f"Rendering cards (pairs={max_pair})...")
    cards = _ensure_cards(timeline, script, Path(cards_dir), theme,
                          channel_name, badge_text, outro_text, max_pair)

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
                    _build_video_block(intro_video, block_segs, out_path)
                else:
                    _build_block(card, block_segs, audio_results, out_path)
            except RuntimeError as e:
                if str(e) == "stopped":
                    raise
                print(f"  [Sleep] Block {bi} failed ({e}), retry once...")
                if is_intro_video:
                    _build_video_block(intro_video, block_segs, out_path)
                else:
                    _build_block(card, block_segs, audio_results, out_path)
        block_paths.append(out_path)
        if bi % 10 == 0 or bi == total - 1:
            _cb(int(2 + bi / total * 78),
                f"Block {bi + 1}/{total} ({t}, {head.get('pair', '')})".strip())

    _cb(82, "Concatenating blocks...")
    no_sub = str(vid_dir / "final_no_sub.mp4")
    concat_segments(block_paths, no_sub, tmp_dir=str(tmp_dir))

    shutil.rmtree(tmp_dir, ignore_errors=True)

    _cb(92, "Final loudnorm...")
    apply_final_loudnorm(no_sub, str(vid_dir))

    yt_title = script.get("youtube_title") or script.get("title") or "sleep_video"
    final_path = vid_dir / f"{safe_filename(yt_title, 'sleep_video')}.mp4"
    os.replace(no_sub, final_path)
    dur = get_duration(str(final_path))
    _cb(100, f"Sleep video done: {final_path.name} ({dur / 60:.1f} min)")
    return str(final_path)
