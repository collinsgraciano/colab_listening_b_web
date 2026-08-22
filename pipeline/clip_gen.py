"""Video clip generation: batch creation, polling, retry, and task builders."""
import json
import os
import time

from mcp_client import call_tool, parse_task_id, poll_task, download_file
from media_utils import get_duration as _get_audio_duration
from grouping_b import merge_group_prompt


def file_ok(path: str, min_size: int) -> bool:
    """Check a file exists and is larger than min_size (guards partial downloads)."""
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > min_size


def build_scene_clip_task(scene: str, scene_url: str) -> dict:
    """Build the clip_0 scene establishing-pan task (pure function)."""
    return {
        "image_urls": scene_url,
        "prompt": f"{scene}, slow camera pan, establishing shot, no characters, 3D cartoon style. The video MUST closely reference the uploaded reference image. CRITICAL: do NOT show any characters in this establishing shot.",
        "filename": "clip_0.mp4",
        "duration": 5,
        "generate_audio": True,
    }


def build_group_clip_tasks(scene: str, char_scene_url: str, groups: list[dict],
                           dialogue: list[dict], pad: float) -> list[dict]:
    """Build one Seedance2 video task per dialogue group (pure function).

    Duration = group total TTS audio + pad per line, rounded to int and
    clamped to the Seedance2 API range 4-15s.
    """
    tasks = []
    for gi, group in enumerate(groups):
        combined_prompt = merge_group_prompt(group, dialogue)
        video_prompt = f"{scene}. {combined_prompt} 3D cartoon style. The video MUST closely reference the uploaded reference image — the characters' appearance, clothing, and the scene must match the reference image exactly. CRITICAL: only ONE instance of each character should appear on screen — do NOT create duplicate characters or clones. Each character appears exactly once, no mirror images, no doubling."
        n_lines_in_group = len(group["lines"])
        group_total_with_pad = group["total_audio"] + n_lines_in_group * pad
        group_dur = round(group_total_with_pad)
        group_dur = max(4, min(group_dur, 15))
        tasks.append({
            "image_urls": char_scene_url,
            "prompt": video_prompt,
            "filename": f"clip_{gi+1}.mp4",
            "duration": group_dur,
            "generate_audio": False,
        })
    return tasks


def create_and_poll_clip(task, clips_dir, idx, total, stop_check=None):
    """Create a single video task, poll it, and download the result.

    Returns the local path on success, or None on failure.
    """
    if stop_check and stop_check():
        return None
    print(f"  [Video] Creating task {idx+1}/{total}: {task['filename']}...")
    try:
        result = call_tool("generate_video", {
            "mode": "reference_image",
            "image_urls": task["image_urls"],
            "prompt": task["prompt"],
            "duration": task["duration"],
            "ratio": "16:9",
            "resolution": "720p",
            "generate_audio": task.get("generate_audio", False),
        })
        task_id = parse_task_id(result)
        if not task_id:
            print(f"    [Video] WARNING: no task_id for {task['filename']}")
            if result and "result" in result:
                print(f"    [Video] API response content: {json.dumps(result['result'].get('content', []), ensure_ascii=False)[:1000]}")
            elif result:
                print(f"    [Video] Full API response: {json.dumps(result, ensure_ascii=False)[:1000]}")
            return None
    except Exception as e:
        print(f"    [Video] ERROR creating task: {e}")
        return None

    if stop_check and stop_check():
        print(f"  [Video] Stop requested, skipping poll for {task['filename']}")
        return None
    print(f"  [Video] Polling: {task['filename']}...")
    data = poll_task(task_id, interval=40, stop_check=stop_check)
    if data.get("status") == "stopped":
        print(f"  [Video] Stopped during poll: {task['filename']}")
        return None
    url = data.get("url", "")
    if url:
        dest = str(clips_dir / task["filename"])
        if download_file(url, dest) and file_ok(dest, 500000):
            print(f"    [Video] Downloaded: {task['filename']} ({os.path.getsize(dest)//1024}KB)")
            return dest
        else:
            print(f"    [Video] WARNING: File too small or download failed, re-downloading...")
            if download_file(url, dest) and file_ok(dest, 500000):
                return dest
            else:
                print(f"    [Video] ERROR: Downloaded file still too small/missing")
                return None
    else:
        print(f"    [Video] ERROR: No video URL. Status={data.get('status')}")
        raw_json = data.get("raw_json", "")
        raw_text = data.get("raw_text", "")
        error_msg = data.get("error", "")
        if error_msg:
            print(f"    [Video] Error message: {error_msg}")
        if raw_json:
            print(f"    [Video] Raw resource JSON: {raw_json[:1000]}")
        if raw_text and not raw_json:
            print(f"    [Video] Raw text block: {raw_text[:1000]}")
        return None


def retry_clip(task, clips_dir, idx, max_retries=5, stop_check=None):
    """Retry a single failed clip by creating brand-new video tasks.

    Each retry creates a completely new generate_video request (not re-querying
    the old task_id). Returns the local path on success, or None after exhausting retries.
    """
    for attempt in range(1, max_retries + 1):
        if stop_check and stop_check():
            return None
        print(f"  [Video] Retrying clip {idx+1} ({task['filename']}) attempt {attempt}/{max_retries}...")
        try:
            result = call_tool("generate_video", {
                "mode": "reference_image",
                "image_urls": task["image_urls"],
                "prompt": task["prompt"],
                "duration": task["duration"],
                "ratio": "16:9",
                "resolution": "720p",
                "generate_audio": task.get("generate_audio", False),
            })
            task_id = parse_task_id(result)
            if not task_id:
                print(f"    [Video] FAILED: could not create task (parse_task_id returned empty)")
                if result and "result" in result:
                    raw_content = json.dumps(result["result"].get("content", []), ensure_ascii=False)[:1500]
                    print(f"    [Video] API response content: {raw_content}")
                elif result:
                    print(f"    [Video] Full API response: {json.dumps(result, ensure_ascii=False)[:1500]}")
                time.sleep(10)
                continue
            data = poll_task(task_id, interval=40, stop_check=stop_check)
            if data.get("status") == "stopped":
                print(f"  [Video] Stopped during retry poll: {task['filename']}")
                return None
            url = data.get("url", "")
            if not url:
                print(f"    [Video] FAILED: task {task_id[:16]}... returned no URL. Status={data.get('status')}")
                raw_json = data.get("raw_json", "")
                raw_text = data.get("raw_text", "")
                raw_response = data.get("raw_response", "")
                error_msg = data.get("error", "")
                if error_msg:
                    print(f"    [Video] Error message: {error_msg}")
                if raw_json:
                    print(f"    [Video] Raw resource JSON: {raw_json[:1000]}")
                if raw_text and not raw_json:
                    print(f"    [Video] Raw text block: {raw_text[:1000]}")
                if raw_response and not raw_json and not raw_text:
                    print(f"    [Video] Full check_task response: {raw_response[:1000]}")
                time.sleep(10)
                continue
            dest = str(clips_dir / task["filename"])
            if download_file(url, dest) and file_ok(dest, 500000):
                print(f"    [Video] Retry successful: {task['filename']}")
                return dest
            else:
                print(f"    [Video] FAILED: file too small or missing, creating new task...")
                time.sleep(10)
                continue
        except Exception as e:
            print(f"    [Video] FAILED (attempt {attempt}): {type(e).__name__}: {e}")
            time.sleep(10)
            continue

    print(f"    [Video] GIVING UP on clip {idx+1} after {max_retries} retries: {task['filename']}")
    return None


def generate_video_clips(video_tasks, clips_dir, clip_paths, offset=0,
                         stop_check=None, max_concurrency=4):
    """Generate video clips with concurrent task creation and polling. Runs in a thread.

    Each task has its own 'duration' (dynamic per-group, not fixed).
    clip_paths[offset + i] corresponds to video_tasks[i]; entries already set
    (per-clip resume) are skipped so no credits are re-spent.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(video_tasks)

    # Collect pending indices (skip already-existing clips)
    pending = []
    for i in range(total):
        idx = offset + i
        if clip_paths[idx] is not None:
            continue
        pending.append(idx)

    if pending:
        print(f"  [Video] Generating {len(pending)} clips with {max_concurrency} concurrent tasks...")
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {}
            for idx in pending:
                if stop_check and stop_check():
                    break
                task = video_tasks[idx - offset]
                future = pool.submit(create_and_poll_clip, task, clips_dir, idx, total, stop_check)
                futures[future] = idx
                # Small delay between submissions to avoid MCP API rate limiting
                time.sleep(3)

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    dest = future.result()
                    if dest:
                        clip_paths[idx] = dest
                except Exception as e:
                    print(f"  [Video] ERROR for clip {idx+1}: {e}")

    if stop_check and stop_check():
        ok_count = sum(1 for p in clip_paths[offset:offset + total] if p is not None)
        print(f"  [Video] Stopped. Clips for this batch: {ok_count}/{total}")
        return

    # Retry failed clips (sequential, one by one)
    failed = [offset + i for i in range(total) if clip_paths[offset + i] is None]
    for idx in failed:
        if stop_check and stop_check():
            break
        task = video_tasks[idx - offset]
        dest = retry_clip(task, clips_dir, idx, stop_check=stop_check)
        if dest:
            clip_paths[idx] = dest

    ok_count = sum(1 for p in clip_paths[offset:offset + total] if p is not None)
    print(f"  [Video] Clips for this batch: {ok_count}/{total}")
