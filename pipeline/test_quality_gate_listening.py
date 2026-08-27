"""Listening 质检门禁（quality_gate_listening）单元测试。

三类用例：
1. 干净夹具（18 行，各字段/一致性达标）→ passed=True，0 error
2. 缺陷夹具（逐项注入缺陷）→ 精确断言检出（重复/性别/空字段/简体/场景/非法speaker/行数/引语）
3. 集成回归：output/ 下真实 listening script.json（若存在）→ 报告形状检查，无则 skip

运行：python test_quality_gate_listening.py
"""
import json
import sys
import unittest
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from quality_gate_listening import run_listening_quality_gate

CA_DESC = "a young woman with brown hair in a green apron"
CB_DESC = "a tall man with short black hair in a blue shirt"

# (speaker, text, phonetic, zh) — 18 行 A2 咖啡店对话，交替发言
_LINES = [
    ("char_a", "Well, this coffee shop is really busy today.",
     "/wɛl/", "嗯，這間咖啡店今天真的很忙。"),
    ("char_b", "Yeah, the morning rush never really stops here.",
     "/jɛə/", "是啊，每天早上這裡都是大排長龍。"),
    ("char_a", "I would like a medium latte, please.",
     "/aɪ wʊd laɪk/", "我想要一杯中杯拿鐵，麻煩你。"),
    ("char_b", "Sure. Do you want that hot or iced?",
     "/ʃʊr/", "好的。你想要熱的還是冰的？"),
    ("char_a", "Oh, iced sounds perfect right now, actually.",
     "/oʊ aɪst/", "哦，冰的現在聽起來正合適。"),
    ("char_b", "Great choice. Any milk or sugar with that?",
     "/ɡreɪt tʃɔɪs/", "好選擇。要加牛奶或糖嗎？"),
    ("char_a", "Just a little oat milk, you know, for taste.",
     "/dʒʌst/", "只要一點燕麥奶，你知道，調調味。"),
    ("char_b", "No problem. That comes to four fifty.",
     "/noʊ ˈprɑbləm/", "沒問題。一共是四塊五。"),
    ("char_a", "Here you go. Um, can I get a receipt?",
     "/hɪr ju ɡoʊ/", "給你。嗯，可以給我一張收據嗎？"),
    ("char_b", "Of course, one second. Oh, our pastry case just came out fresh.",
     "/əv kɔrs/", "當然，等一下。哦，我們的點心剛出爐。"),
    ("char_a", "Wow, that croissant smells amazing, I might add one.",
     "/waʊ/", "哇，那個可頌聞起來好香，我可能加一個。"),
    ("char_b", "Honestly, they are the best seller in this shop.",
     "/ˈɑnɪstli/", "說真的，它們是這家店裡的銷量冠軍。"),
    ("char_a", "Actually, make it two croissants then, please.",
     "/ˈæktʃuəli/", "那其實請給我兩個可頌。"),
    ("char_b", "You got it. Your order will be right up.",
     "/ju ɡɑt ɪt/", "你拿到了。你的餐點馬上就好。"),
    ("char_a", "That makes sense why everyone grabs one here.",
     "/ðæt meɪks sɛns/", "難怪這裡每個人都買一個。"),
    ("char_b", "We bake them every morning before opening.",
     "/wi beɪk/", "我們每天開店前現烤。"),
    ("char_a", "Well, I am definitely coming back tomorrow.",
     "/wɛl aɪ æm/", "嗯，我明天一定還會再來。"),
    ("char_b", "Awesome, see you then. Have a great day!",
     "/ˈɔsəm/", "太好了，到時見。祝你今天愉快！"),
]


def _prompts(sp: str, text: str):
    desc = CA_DESC if sp == "char_a" else CB_DESC
    role = "the customer" if sp == "char_a" else "the barista"
    image = f"{desc}, {role}, in the coffee shop, acting out the line naturally"
    video = (f"{desc} in the coffee shop, saying: \"{text}\" while matching "
             f"the reference image exactly")
    poses = [f"{desc}, speaking with mouth open, gesturing",
             f"{desc}, listening with a slight smile, relaxed"]
    return image, video, poses


def _clean_script() -> dict:
    dialogue = []
    for i, (sp, text, ph, zh) in enumerate(_LINES):
        img, vid, poses = _prompts(sp, text)
        dialogue.append({"speaker": sp, "text": text, "phonetic": ph, "zh": zh,
                         "image_prompt": img, "video_prompt": vid,
                         "poses": poses})
    return {
        "lesson_type": "listening",
        "title": "AT THE COFFEE SHOP",
        "title_zh": "咖啡店",
        "title_quote": "A medium latte, please",
        "scene": "coffee shop",
        "scene_zh": "咖啡店 · 點餐",
        "cefr": "A2",
        "welcome_en": "Hi friends! Welcome back! Today we are ordering at a busy coffee shop.",
        "welcome_zh": "嗨朋友們！歡迎回來！今天我們在忙碌的咖啡店點餐。",
        "outro": "That's all for today. Keep practicing!",
        "outro_zh": "今天就到這裡。繼續練習！",
        "thumbnail_prompt": "a cheerful young woman with an excited face at a coffee shop counter, bright colors",
        "youtube_title": ("【沉浸式英文動畫】咖啡店點餐 ☕ 18句超實用生活英文，"
                          "聽完就能開口說！每天聽一集，三個月開口不尷尬，"
                          "不用背單字（出國旅遊也用得到）｜Coffee Shop Ordering"),
        "youtube_tags": [f"tag{i}" for i in range(15)],
        "thumbnail_icons": [{"en": "Latte", "zh": "拿鐵"}, {"en": "Iced", "zh": "冰飲"},
                            {"en": "Croissant", "zh": "可頌"}, {"en": "Receipt", "zh": "收據"}],
        "char_a_description": CA_DESC,
        "char_b_description": CB_DESC,
        "char_a_gender": "female",
        "char_b_gender": "male",
        "char_a_role": "customer",
        "char_b_role": "barista",
        "dialogue": dialogue,
        "_requested_num_lines": 18,
    }


def _find_issues(report: dict, check: str = None, severity: str = None,
                 detail_sub: str = None) -> list[dict]:
    out = []
    for i in report["issues"]:
        if check and i["check"] != check:
            continue
        if severity and i["severity"] != severity:
            continue
        if detail_sub and detail_sub not in i["detail"]:
            continue
        out.append(i)
    return out


class TestCleanFixture(unittest.TestCase):
    def test_clean_passes(self):
        report = run_listening_quality_gate(_clean_script(), 18)
        self.assertTrue(report["passed"],
                        f"clean fixture should pass, issues={report['issues']}")
        self.assertEqual(report["n_errors"], 0)


class TestDefectFixture(unittest.TestCase):
    def _defect(self, mutate) -> dict:
        script = _clean_script()
        mutate(script)
        return script

    def test_line_count(self):
        report = run_listening_quality_gate(_clean_script(), 19)
        found = _find_issues(report, "structure", "error", "总行数")
        self.assertTrue(found, "line-count mismatch should be an error")

    def test_duplicate_line(self):
        def mutate(s):
            s["dialogue"][5]["text"] = s["dialogue"][2]["text"]
        report = run_listening_quality_gate(self._defect(mutate), 18)
        found = _find_issues(report, "duplicates", "error", "完全重复行")
        self.assertTrue(found)
        self.assertEqual(found[0]["lines"], [2, 5])

    def test_gender_conflict(self):
        def mutate(s):
            s["dialogue"][7]["image_prompt"] = (
                f"{CB_DESC}, the barista, she pours milk in the coffee shop")
        report = run_listening_quality_gate(self._defect(mutate), 18)
        found = _find_issues(report, "consistency", "error", "男女指示词混用")
        self.assertTrue(found)
        self.assertIn(7, found[0]["lines"])

    def test_empty_zh(self):
        def mutate(s):
            s["dialogue"][3]["zh"] = ""
        report = run_listening_quality_gate(self._defect(mutate), 18)
        self.assertTrue(_find_issues(report, "fields", "error", "'zh' 为空"))

    def test_empty_image_prompt(self):
        def mutate(s):
            s["dialogue"][4]["image_prompt"] = ""
        report = run_listening_quality_gate(self._defect(mutate), 18)
        self.assertTrue(_find_issues(report, "fields", "error", "'image_prompt' 为空"))

    def test_simplified_zh(self):
        def mutate(s):
            s["dialogue"][4]["zh"] = "我现在要一杯拿铁"
        report = run_listening_quality_gate(self._defect(mutate), 18)
        found = _find_issues(report, "translation", "warning", "简体")
        self.assertTrue(found)
        self.assertIn(4, found[0]["lines"])

    def test_scene_drift(self):
        def mutate(s):
            for i in range(9, 15):
                s["dialogue"][i]["image_prompt"] = f"a person at the airport terminal, line {i}"
                s["dialogue"][i]["video_prompt"] = f"a person at the airport terminal, line {i}"
        report = run_listening_quality_gate(self._defect(mutate), 18)
        found = _find_issues(report, "consistency", "warning", "未提及")
        self.assertTrue(found, "33% scene-missing lines should trigger scene warning")
        self.assertIn(9, found[0]["lines"])

    def test_bad_speaker(self):
        def mutate(s):
            s["dialogue"][8]["speaker"] = "narrator"
        report = run_listening_quality_gate(self._defect(mutate), 18)
        found = _find_issues(report, "structure", "error", "speaker 非法")
        self.assertTrue(found)
        self.assertIn(8, found[0]["lines"])

    def test_phonetic_format(self):
        def mutate(s):
            s["dialogue"][2]["phonetic"] = "hello world"
        report = run_listening_quality_gate(self._defect(mutate), 18)
        self.assertTrue(_find_issues(report, "fields", "warning", "/slashes/"))

    def test_poses_missing(self):
        def mutate(s):
            s["dialogue"][2]["poses"] = []
        report = run_listening_quality_gate(self._defect(mutate), 18)
        self.assertTrue(_find_issues(report, "fields", "warning", "poses"))

    def test_title_quote_mismatch(self):
        def mutate(s):
            s["title_quote"] = "Give me a burger"
        report = run_listening_quality_gate(self._defect(mutate), 18)
        self.assertTrue(_find_issues(report, "consistency", "warning", "未逐字命中"))


class TestMergeAndFixes(unittest.TestCase):
    """llm_review 的纯 Python 部分（不触发任何 LLM 调用）。"""

    @classmethod
    def setUpClass(cls):
        from llm_review import _apply_fixes, _merge_patches
        cls._apply_fixes = staticmethod(_apply_fixes)
        cls._merge_patches = staticmethod(_merge_patches)

    def test_apply_fixes_clamps_speaker(self):
        script = _clean_script()
        script["dialogue"][0]["speaker"] = "narrator"
        script["dialogue"][1]["speaker"] = ""
        self._apply_fixes(script)
        self.assertEqual(script["dialogue"][0]["speaker"], "char_a")
        self.assertEqual(script["dialogue"][1]["speaker"], "char_a")

    def test_merge_patches(self):
        report = {"issues": [
            {"check": "duplicates", "severity": "error",
             "detail": "完全重复行 line 2 == line 5: \"...\"", "lines": [2, 5]},
        ]}
        judge_issues = [{"type": "story", "lines": [10, 12],
                         "problem": "facts contradict", "fix_hint": "unify the price"}]
        patches = self._merge_patches(report, judge_issues, _clean_script())
        self.assertTrue(patches)
        dup = [p for p in patches if p[0] <= 4 and p[1] >= 5]
        self.assertTrue(dup, "duplicate at 2/5 should merge into a patch spanning 4-6")
        story = [p for p in patches if p[0] <= 10 and p[1] >= 12]
        self.assertTrue(story, "judge issue 10-12 should become a patch")
        for s, e, hints in patches:
            self.assertLessEqual(e - s + 1, 20)
            self.assertTrue(hints)

    def test_merge_patches_ignores_warnings(self):
        report = {"issues": [
            {"check": "naturalness", "severity": "warning",
             "detail": "平均每行 12.0 词", "lines": []},
        ]}
        patches = self._merge_patches(report, [], _clean_script())
        self.assertEqual(patches, [])


def _find_real_scripts() -> list[dict]:
    root = Path(__file__).resolve().parent.parent
    found = []
    out_dir = root / "output"
    if out_dir.exists():
        for f in out_dir.rglob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                s = data.get("script", data)
                if isinstance(s, dict) and s.get("lesson_type") == "listening" \
                        and s.get("dialogue"):
                    found.append(s)
            except Exception:
                continue
    return found


class TestIntegration(unittest.TestCase):
    def test_real_script_report_shape(self):
        scripts = _find_real_scripts()
        if not scripts:
            self.skipTest("output/ 下没有真实 listening script.json")
        report = run_listening_quality_gate(scripts[0])
        self.assertIn("passed", report)
        self.assertIn("issues", report)
        self.assertIn("summary", report)
        self.assertIsInstance(report["issues"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
