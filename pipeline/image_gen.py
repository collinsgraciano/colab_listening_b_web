"""Image generation: character/scene images, per-line dialogue images, resume check."""
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

from mcp_client import call_tool, parse_task_id, poll_task, download_file
from checkpoint import step_done
from media_utils import get_duration as _get_audio_duration
from atlas_split import split_atlas
from style_manager import DEFAULT_STYLE_PROMPT


def check_step2_resume(checkpoint, script, dirs, n, is_quest):
    """Check if Step 2 can be resumed from existing files. Returns (tts_results, image_urls) or None.
    """
    img_dir, audio_dir = dirs["images"], dirs["audio"]
    step2_done = step_done(checkpoint, "step2_images_tts")
    char_scene_file = img_dir / "char_scene.png"
    # Quest mode uses scene_0.png instead of scene.png; pose_char_a_0.png instead of char_scene.png
    scene_file = img_dir / "scene_0.png" if is_quest else img_dir / "scene.png"
    ref_file = img_dir / "pose_char_a_0.png" if is_quest else char_scene_file
    all_audio_exist = all((audio_dir / f"dialogue_{i}.mp3").exists() for i in range(n))
    if is_quest:
        all_zh_exist = True
    else:
        all_zh_exist = all((audio_dir / f"zh_{i}.mp3").exists() for i in range(n))
    # 旁白音频清单按结构区分：original_cutout 主持人开场沿用 quest 三段 + practice_intro
    struct_val = checkpoint.get("structure", "")
    if struct_val == "original_cutout":
        narration_names = ["welcome", "hook", "outro", "practice_intro"]
    elif is_quest:
        narration_names = ["welcome", "hook", "outro"]
    else:
        narration_names = ["intro", "outro", "practice_intro"]
    narration_files = [f"{name}.mp3" for name in narration_names]
    narration_exist = all((audio_dir / f).exists() for f in narration_files)
    # Only original_static generates per-line dialogue images
    needs_dialogue_imgs = (checkpoint.get("structure", "") == "original_static")
    all_dialogue_imgs_exist = (not needs_dialogue_imgs) or all(
        (img_dir / f"dialogue_img_{i}.png").exists() for i in range(n))

    if not (step2_done and ref_file.exists() and scene_file.exists()
            and all_audio_exist and all_zh_exist
            and narration_exist and all_dialogue_imgs_exist):
        return None

    print("  [Resume] Step 2 already done, loading existing images + audio...")
    # Quest mode re-uploads pose_char_a_0.png + scene_0.png; others re-upload char_scene.png + scene.png
    reupload_files = ["pose_char_a_0.png", "scene_0.png"] if is_quest else ["char_scene.png", "scene.png"]
    image_urls = {}
    for filename in reupload_files:
        filepath = str(img_dir / filename)
        url = reupload_for_cdn(filepath, filename)
        if url:
            image_urls[filename] = url

    normal_paths = [str(audio_dir / f"dialogue_{i}.mp3") for i in range(n)]
    if is_quest:
        zh_paths = [""] * n
    else:
        zh_paths = [str(audio_dir / f"zh_{i}.mp3") for i in range(n)]
    dialogue_durations = [_get_audio_duration(p) for p in normal_paths]
    narration = {}
    for name in narration_names:
        narration[name] = str(audio_dir / f"{name}.mp3")
    tts_results = {
        "narration": narration,
        "normal_paths": normal_paths,
        "dialogue_durations": dialogue_durations,
        "zh_paths": zh_paths,
        "vocab_paths": [],
        "slow_paths": [],
        "slow_durations": [],
        "quiz_paths": [],
    }
    print("  [Resume] Images + audio loaded.")
    return tts_results, image_urls


def reupload_for_cdn(filepath, filename):
    """Re-upload an existing local image to TOS and return the CDN URL.

    file_upload MCP tool only returns presigned URLs — the actual PUT upload
    must be done by the caller. Without this, Seedance2 gets a URL pointing to
    nothing → 'resource download failed'.
    """
    print(f"  [Image] {filename} already exists, re-uploading for CDN URL...")
    try:
        upload_result = call_tool("file_upload", {"file_path": filepath})
        if "result" not in upload_result:
            print(f"    [Image] file_upload returned no result")
            return ""
        upload_url = ""
        file_url = ""
        for item in upload_result["result"].get("content", []):
            if item.get("type") == "resource":
                res_json = json.loads(item.get("resource", {}).get("text", ""))
                upload_url = res_json.get("upload_url", "")
                file_url = res_json.get("file_url", "")
                break
            elif item.get("type") == "text":
                m = re.search(r"(https?://[^\s`'\")]+)", item.get("text", ""))
                if m and not file_url:
                    file_url = m.group(1)
        if not file_url:
            print(f"    [Image] No file_url in file_upload response")
            return ""
        if not upload_url:
            print(f"    [Image] WARNING: No upload_url — file may already exist on TOS")
        else:
            # PUT the actual file to upload_url
            print(f"    [Image] Uploading {filename} to TOS...")
            with open(filepath, "rb") as f:
                file_data = f.read()
            req = urllib.request.Request(upload_url, data=file_data, method="PUT")
            req.add_header("Content-Type", "image/png")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    if resp.status == 200:
                        print(f"    [Image] Upload OK ({len(file_data)//1024}KB)")
                    else:
                        print(f"    [Image] Upload returned status {resp.status}")
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")[:300]
                print(f"    [Image] Upload FAILED: HTTP {e.code}: {body}")
                return ""
        print(f"    [Image] CDN URL: {file_url[:80]}...")
        return file_url
    except Exception as e:
        print(f"    [Image] Re-upload failed: {e}")
    return ""


def generate_images(image_prompts, img_dir, tts_thread, max_workers=4,
                    image_size="landscape_16_9"):
    """Generate character/scene images via MCP (concurrent). Returns image_urls dict.

    image_size: MCP generate_image size spec — "landscape_16_9" (default) or
    "portrait_16_9" (9:16 vertical).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    image_urls = {}
    image_failed = False

    # Filter out already-existing files (but re-upload them for CDN URLs)
    pending = [(prompt, filename) for prompt, filename in image_prompts
               if not os.path.exists(str(img_dir / filename))]
    for _, filename in image_prompts:
        if os.path.exists(str(img_dir / filename)):
            print(f"  [Image] {filename} already exists, re-uploading for CDN URL...")
            url = reupload_for_cdn(str(img_dir / filename), filename)
            if url:
                image_urls[filename] = url

    if not pending:
        return image_urls

    print(f"  [Image] Generating {len(pending)} images ({max_workers} concurrent)...")

    def _gen_one(prompt, filename):
        print(f"  [Image] Generating: {filename}...")
        try:
            result = call_tool("generate_image", {
                "prompt": prompt,
                "provider": "seedream",
                "image_size": image_size,
                "output_format": "png",
            })
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if url:
                dest = str(img_dir / filename)
                download_file(url, dest)
                print(f"    [Image] Downloaded: {dest}")
                return filename, url, None
            else:
                print(f"    [Image] FATAL: No URL for {filename}")
                return filename, "", "no_url"
        except RuntimeError as e:
            if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                return filename, "", "tokens_exhausted"
            return filename, "", str(e)
        except Exception as e:
            return filename, "", str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_gen_one, prompt, filename): filename
                for prompt, filename in pending}
        for fut in as_completed(futs):
            filename, url, err = fut.result()
            if err == "tokens_exhausted":
                print("\n  [FATAL] 所有 MCP Token 积分已耗尽！请充值后重新运行（--resume）继续。")
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            if url:
                image_urls[filename] = url
            elif err:
                image_failed = True

    if image_failed:
        missing = [fn for fn, _ in image_prompts if fn not in image_urls]
        print(f"  [Image] ABORTING: Missing required images: {missing}")
        if tts_thread:
            tts_thread.join(timeout=5)
        sys.exit(1)
    return image_urls


def generate_dialogue_images(dialogue, img_dir, char_a_desc, char_b_desc, scene,
                               is_quest, char_scene_cdn, char_scene_c_cdn,
                               tts_thread, max_workers=4,
                               style_prompt: str = DEFAULT_STYLE_PROMPT,
                               image_size=None):
    """Generate per-line dialogue images for static/quest modes (5 concurrent).

    image_size: size dict for MCP generate_image, e.g. {"width":1280,"height":720}
    (default landscape) or {"width":720,"height":1280} (9:16 vertical).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = len(dialogue)
    mode_label = "quest" if is_quest else "original_static"
    print(f"  [Image] Generating {n} dialogue line images ({mode_label} mode, {max_workers} concurrent)...")
    if image_size is None:
        image_size = {"width": 1280, "height": 720}

    def _gen_one(i, line):
        d_img_path = str(img_dir / f"dialogue_img_{i}.png")
        if os.path.exists(d_img_path):
            print(f"    [Image] dialogue_img_{i} already exists, skipping")
            return i, True
        img_prompt = line.get("image_prompt",
                                f"{char_a_desc} and {char_b_desc} at {scene}, {style_prompt}, 16:9")
        if is_quest and line.get("phase") == "core" and char_scene_c_cdn:
            ref_cdn = char_scene_c_cdn
        else:
            ref_cdn = char_scene_cdn
        print(f"    [Image] Generating dialogue_img_{i}/{n}...")
        try:
            gen_params = {
                "prompt": img_prompt,
                "provider": "frontier",
                "quality": "high",
                "image_size": image_size,
                "output_format": "png",
            }
            if ref_cdn:
                gen_params["image_urls"] = ref_cdn
            result = call_tool("generate_image", gen_params)
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if url:
                download_file(url, d_img_path)
                print(f"      [Image] Downloaded: dialogue_img_{i}.png")
                return i, True
            print(f"      [Image] WARNING: No URL for dialogue_img_{i}, will use scene image as fallback")
            return i, False
        except RuntimeError as e:
            if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                print("\n  [FATAL] 所有 MCP Token 积分已耗尽！请充值后重新运行（--resume）继续。")
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            print(f"      [Image] ERROR generating dialogue_img_{i}: {e}")
            return i, False
        except Exception as e:
            print(f"      [Image] ERROR generating dialogue_img_{i}: {e}")
            return i, False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_gen_one, i, line) for i, line in enumerate(dialogue)]
        for fut in as_completed(futs):
            fut.result()


def generate_pose_images(dialogue, img_dir, char_a_desc, char_b_desc, scene,
                          char_a_ref_cdn, char_b_ref_cdn, tts_thread, max_workers=4,
                          style_prompt: str = DEFAULT_STYLE_PROMPT):
    """Generate per-line character pose atlas (2×2 grid) and split into 4 poses.

    For each dialogue line, generates ONE image containing 4 poses arranged in
    a 2×2 grid (top-left: speaking, top-right: listening, bottom-left:
    thinking, bottom-right: reacting). The image is then split into 4 separate
    pose_{i}_{j}.png files. This guarantees character consistency across poses
    and reduces API calls by 4×.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = len(dialogue)
    print(f"  [PoseAtlas] Generating {n} pose atlases (2×2 grid → 4 poses each)...")

    def _gen_one(line_idx, line):
        # Check if split poses already exist
        all_exist = all(
            os.path.exists(str(img_dir / f"pose_{line_idx}_{j}.png"))
            for j in range(4)
        )
        if all_exist:
            print(f"    [PoseAtlas] line {line_idx} poses already exist, skipping")
            return (line_idx, True)

        atlas_path = str(img_dir / f"pose_atlas_{line_idx}.png")
        poses = line.get("poses", [])
        if not poses:
            img_prompt = line.get("image_prompt", f"{char_a_desc} at {scene}")
            poses = [img_prompt, img_prompt]

        # Pick the correct character reference based on speaker
        speaker = line.get("speaker", "char_a")
        if speaker == "char_b":
            ref_cdn = char_b_ref_cdn
            char_desc = char_b_desc
        else:
            ref_cdn = char_a_ref_cdn
            char_desc = char_a_desc

        # Build 2×2 atlas prompt: 4 poses in one image
        pose_a = poses[0] if len(poses) > 0 else "speaking with mouth open"
        pose_b = poses[1] if len(poses) > 1 else "listening with slight smile"
        # Pad to 4 poses with variations
        pose_c = "thinking with hand on chin"
        pose_d = "surprised with raised eyebrows"

        # Ensure character description is in each pose
        def _with_desc(p):
            if char_desc and char_desc.lower() not in p.lower():
                return f"{char_desc}, {p}"
            return p

        atlas_prompt = (
            f"2×2 grid character pose sheet, four poses of the same character, "
            f"top-left: {_with_desc(pose_a)}, "
            f"top-right: {_with_desc(pose_b)}, "
            f"bottom-left: {_with_desc(pose_c)}, "
            f"bottom-right: {_with_desc(pose_d)}, "
            f"half-body close-up, waist up, all four poses same character same outfit, "
            f"plain white background, {style_prompt}, "
            f"clean outline tightly hugging the character silhouette, "
            f"no props, no objects, no scene, no text"
        )

        print(f"    [PoseAtlas] Generating atlas for line {line_idx}...")
        try:
            gen_params = {
                "prompt": atlas_prompt,
                "provider": "seedream",
                "image_size": "4096x4096",
                "output_format": "png",
                "is_segmentation": True,
            }
            if ref_cdn:
                gen_params["image_urls"] = ref_cdn
            result = call_tool("generate_image", gen_params)
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if not url:
                print(f"      [PoseAtlas] WARNING: No URL for line {line_idx}")
                return (line_idx, False)

            download_file(url, atlas_path)
            print(f"      [PoseAtlas] Downloaded atlas: pose_atlas_{line_idx}.png")

            # Split into 4 quadrants（含分隔线检测 + 残边修剪，无缝时回退等分）
            out_paths = [str(img_dir / f"pose_{line_idx}_{j}.png") for j in range(4)]
            split_atlas(atlas_path, 2, 2, out_paths, log_prefix="[PoseAtlas]")

            # atlas 原图保留，不再删除

            return (line_idx, True)
        except RuntimeError as e:
            if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                print("\n  [FATAL] 所有 MCP Token 积分已耗尽！")
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            print(f"      [PoseAtlas] ERROR line {line_idx}: {e}")
            return (line_idx, False)
        except Exception as e:
            print(f"      [PoseAtlas] ERROR line {line_idx}: {e}")
            return (line_idx, False)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_gen_one, i, line) for i, line in enumerate(dialogue)]
        for fut in as_completed(futs):
            fut.result()

    print(f"  [PoseAtlas] Done — {n} atlases → {n*4} pose images.")


def generate_quest_atlases(script, img_dir, tts_thread, max_workers=4,
                           style_prompt: str = DEFAULT_STYLE_PROMPT,
                           char_keys=None):
    """Generate character pose atlases for quest mode.

    All characters (char_a, char_b, char_c, host): 4×2 grid (8 poses each).
    Shared style prefix ensures visual consistency across all characters.
    If char_keys is provided, only those characters are generated (e.g. original_cutout mode).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _STYLE = style_prompt

    all_chars = [
        ("char_a", script.get("char_a_description", "friendly young man"), 8),
        ("char_b", script.get("char_b_description", "friendly young woman"), 8),
        ("char_c", script.get("char_c_description", "friendly staff member"), 8),
        ("host", script.get("host_description", "friendly young woman with short brown hair, wearing a smart blue blazer, warm smile, professional TV host appearance"), 8),
    ]
    chars = [c for c in all_chars if char_keys is None or c[0] in char_keys]

    # Generate atlases directly from text (no separate ref images needed)
    def _gen_one_char(char_key, char_desc, n_poses):
        all_exist = all(
            os.path.exists(str(img_dir / f"pose_{char_key}_{j}.png"))
            for j in range(n_poses)
        )
        if all_exist:
            print(f"  [QuestAtlas] {char_key} poses already exist, skipping")
            return

        atlas_path = str(img_dir / f"pose_atlas_{char_key}.png")
        atlas_prompt = (
            f"4x2 grid character pose sheet, eight poses of the same character, "
            f"{char_desc}, "
            f"top row left to right: speaking with mouth open, listening with slight smile, "
            f"thinking with hand on chin, surprised with raised eyebrows, "
            f"bottom row left to right: nodding in agreement, waving right hand, "
            f"pointing forward, laughing with eyes closed, "
            f"half-body close-up, waist up, all eight poses same character same outfit, "
            f"plain white background, {_STYLE}, "
            f"no props, no objects, no scene, no text"
        )
        grid_w, grid_h = 4, 2
        img_size = "4992x3328"

        print(f"  [QuestAtlas] Generating {grid_w}×{grid_h} atlas for {char_key} ({n_poses} poses)...")
        try:
            gen_params = {
                "prompt": atlas_prompt,
                "provider": "seedream",
                "image_size": img_size,
                "output_format": "png",
                "is_segmentation": True,
            }
            result = call_tool("generate_image", gen_params)
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if not url:
                import json as _json
                raw = data.get("raw_json", "")
                if raw:
                    try:
                        rj = _json.loads(raw)
                        out = rj.get("output") or {}
                        d = out.get("data") or {}
                        print(f"    [QuestAtlas] DEBUG {char_key} output.data keys={list(d.keys())}")
                        print(f"    [QuestAtlas] DEBUG {char_key} output.data={_json.dumps(d, ensure_ascii=False)[:600]}")
                        res = d.get("result") or {}
                        print(f"    [QuestAtlas] DEBUG {char_key} result={_json.dumps(res, ensure_ascii=False)[:400]}")
                    except Exception:
                        pass
                print(f"    [QuestAtlas] WARNING: No URL for {char_key}")
                return
            download_file(url, atlas_path)
            print(f"    [QuestAtlas] Downloaded: pose_atlas_{char_key}.png")

            # 切分 4×2 网格（含分隔线检测 + 残边修剪，无缝时回退等分）
            out_paths = [str(img_dir / f"pose_{char_key}_{j}.png")
                         for j in range(grid_w * grid_h)]
            split_atlas(atlas_path, grid_w, grid_h, out_paths, log_prefix="[QuestAtlas]")

            # atlas 原图保留，不再删除
        except RuntimeError as e:
            if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            print(f"    [QuestAtlas] ERROR {char_key}: {e}")
        except Exception as e:
            print(f"    [QuestAtlas] ERROR {char_key}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_gen_one_char, ck, cd, np) for ck, cd, np in chars]
        for fut in as_completed(futs):
            fut.result()

    n = len(chars)
    print(f"  [QuestAtlas] Done — {n} characters × 8 poses = {n * 8} pose images.")
    return {}


def generate_scene_atlas(scene_images, scene, img_dir, tts_thread,
                         style_prompt: str = DEFAULT_STYLE_PROMPT):
    """Generate scene backgrounds in batches of 4 via 2×2 grid atlases.

    Handles any number of scenes: 1-4 → one atlas, 5-8 → two atlases, etc.
    Each atlas is one API call. Saves (N-ceil(N/4)) API calls vs per-scene.
    Uses frontier 4K: atlas ~2880×2880 → each cell ~1440×1440.
    """
    _STYLE = style_prompt

    n_total = len(scene_images)
    if n_total == 0:
        print("  [SceneAtlas] No scene_images in script, skipping")
        return

    # Check if all scene files already exist
    all_exist = all(
        os.path.exists(str(img_dir / f"scene_{si}.png"))
        for si in range(n_total)
    )
    if all_exist:
        print(f"  [SceneAtlas] All {n_total} scene images already exist, skipping")
        return

    n_batches = (n_total + 3) // 4  # ceil(n_total / 4)
    print(f"  [SceneAtlas] {n_total} scenes → {n_batches} atlas batch(es)")

    for batch_idx in range(n_batches):
        start = batch_idx * 4
        end = min(start + 4, n_total)
        batch = scene_images[start:end]
        n_batch = len(batch)

        # Check if this batch's files already exist
        batch_exist = all(
            os.path.exists(str(img_dir / f"scene_{start + j}.png"))
            for j in range(n_batch)
        )
        if batch_exist:
            print(f"  [SceneAtlas] Batch {batch_idx+1} (scene_{start}..scene_{end-1}) already exist, skipping")
            continue

        # Build scene descriptions for the grid
        scene_descs = []
        for si_data in batch:
            si_prompt = si_data.get("prompt", f"a {scene} interior, 16:9, no people")
            scene_descs.append(si_prompt)

        # Pad to 4 with generic descriptions (filler cells are discarded after split)
        while len(scene_descs) < 4:
            scene_descs.append(f"a {scene} interior, wide shot, no people")

        atlas_prompt = (
            f"2x2 grid scene background sheet, four different views of the same location, "
            f"top-left: {scene_descs[0]}, "
            f"top-right: {scene_descs[1]}, "
            f"bottom-left: {scene_descs[2]}, "
            f"bottom-right: {scene_descs[3]}, "
            f"all four panels show the same {scene} location from different angles, "
            f"wide shots showing key elements, no people, no characters, no text, "
            f"seamless collage, panels placed edge to edge, no borders, no gutters, "
            f"no dividing lines, no frames, no white or gray lines between panels, "
            f"no margins around the outer edges, each panel fills its quadrant fully, "
            f"{_STYLE}"
        )

        atlas_path = str(img_dir / f"scene_atlas_{batch_idx}.png")
        print(f"  [SceneAtlas] Generating batch {batch_idx+1}/{n_batches} "
              f"(scene_{start}..scene_{end-1}, {n_batch} scenes)...")

        try:
            gen_params = {
                "prompt": atlas_prompt,
                "provider": "seedream",
                "image_size": "4992x3328",
                "output_format": "png",
            }
            result = call_tool("generate_image", gen_params)
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if not url:
                import json as _json
                raw = data.get("raw_json", "")
                if raw:
                    try:
                        rj = _json.loads(raw)
                        out = rj.get("output") or {}
                        d = out.get("data") or {}
                        print(f"    [SceneAtlas] DEBUG output.data keys={list(d.keys())}")
                    except Exception:
                        pass
                print(f"    [SceneAtlas] WARNING: No URL for batch {batch_idx+1}")
                continue

            download_file(url, atlas_path)
            print(f"    [SceneAtlas] Downloaded: scene_atlas_{batch_idx}.png")

            # 切分 2×2 网格（分隔线检测 + 残边修剪；填充格丢弃，只切前 n_batch 格）
            out_paths = [str(img_dir / f"scene_{start + j}.png") for j in range(n_batch)]
            split_atlas(atlas_path, 2, 2, out_paths, log_prefix="[SceneAtlas]")

            # atlas 原图保留，不再删除（便于排查切分问题）

        except RuntimeError as e:
            if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            print(f"    [SceneAtlas] ERROR batch {batch_idx+1}: {e}")
        except Exception as e:
            print(f"    [SceneAtlas] ERROR batch {batch_idx+1}: {e}")

    print(f"  [SceneAtlas] Done — {n_total} scene images from {n_batches} atlas batch(es).")
