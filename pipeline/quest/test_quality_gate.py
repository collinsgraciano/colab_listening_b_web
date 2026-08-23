"""quality_gate.py 单元测试 — 纯程序化，无 LLM。

三类用例：
1. 干净夹具（num_lines=20，各指标达标）→ passed=True，0 error
2. 缺陷夹具（故意注入各 category 缺陷）→ 精确断言检出
3. 集成回归：真实《市場講價》script.json（若存在）→ 检出已知生产缺陷

运行：python test_quality_gate.py
"""
import json
import sys
import unittest
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from quality_gate import run_quality_gate, format_report


# ---------------------------------------------------------------------------
# 夹具构造
# ---------------------------------------------------------------------------

def _line(speaker, text, phase, zh, on_screen):
    return {"speaker": speaker, "text": text, "phase": phase,
            "zh": zh, "on_screen": on_screen}


def _hook(n_words):
    """生成恰好 n_words 词的 hook 文本。"""
    base = ("hello everyone welcome back to our slow english listening series "
            "for beginners today our friends will visit a local shop and try "
            "something new together before the story begins here is your "
            "listening challenge listen closely and find the surprising answer")
    words = base.split()
    out = []
    while len(out) < n_words:
        out.extend(words)
    return " ".join(out[:n_words])


def _clean_script():
    """20 行干净脚本：(6,10,2,2)，全部指标达标。"""
    dlg = [
        _line("char_a", "Hey Mina, do you want to try bubble tea after class?", "buildup",
              "嘿 Mina，下課後想去喝泡泡茶嗎？", ["char_a", "char_b"]),
        _line("char_b", "Sure. What is it exactly?", "buildup",
              "當然。它到底是什麼？", ["char_a", "char_b"]),
        _line("char_a", "Well, it is a cold tea drink with milk.", "buildup",
              "嗯，它是加了牛奶的冰茶飲。", ["char_a", "char_b"]),
        _line("char_b", "Hmm, do they have real bubbles in it?", "buildup",
              "嗯，裡面有真正的泡泡嗎？", ["char_a", "char_b"]),
        _line("char_a", "That is a secret! You will see at the shop.", "buildup",
              "這是秘密！到店裡你就知道了。", ["char_a"]),
        _line("char_b", "Okay, actually I am very curious now.", "buildup",
              "好，其實我現在很好奇了。", ["char_a", "char_b"]),
        _line("char_c", "Welcome! This is our menu board today.", "core",
              "歡迎！這是今天 的菜單看板。", []),
        _line("char_a", "Wow, there are so many flavors here.", "core",
              "哇，這裡有好多口味。", ["char_a", "char_c"]),
        _line("char_c", "You can choose your tea and sweetness level.", "core",
              "你可以選茶底和甜度。", ["char_a", "char_c"]),
        _line("char_b", "Oh really? That sounds fun to customize.", "core",
              "噢真的嗎？可以客製化聽起來很有趣。", ["char_b", "char_c"]),
        _line("char_a", "Can I get the classic one with less sugar?", "core",
              "我可以點經典口味少糖嗎？", ["char_a", "char_c"]),
        _line("char_c", "Sure thing, boss. Less sugar it is.", "core",
              "沒問題，老闆。少糖的。", ["char_a", "char_c"]),
        _line("char_b", "You know, the staff here are friendly.", "core",
              "你知道嗎，這裡的店員很友善。", ["char_b"]),
        _line("char_a", "Yeah, and the shop smells like fresh tea.", "core",
              "是啊，而且店裡有新鮮茶香。", []),
        _line("char_b", "They brew the leaves every morning, I think.", "core",
              "我想他們每天早上都會煮茶。", ["char_a", "char_b"]),
        _line("char_a", "Kind of amazing. My drink is ready already.", "core",
              "有點厲害。我的飲料已經好了。", ["char_a", "char_c"]),
        _line("char_b", "So, do you remember the secret question?", "reveal",
              "那麼，你還記得那個秘密問題嗎？", ["char_a", "char_b"]),
        _line("char_a", "Yes! The foam comes from shaking the tea hard.", "reveal",
              "記得！泡沫是用力搖茶做出來的。", ["char_a", "char_b"]),
        _line("char_b", "Right, the shaking makes the foam on top.", "review",
              "對，搖晃讓頂部產生泡沫。", ["char_a", "char_b"]),
        _line("char_a", "I see. Let us come back next week again.", "review",
              "明白了。我們下週再來吧。", ["char_a", "char_b"]),
    ]
    return {
        "cefr": "A2",
        "listening_question_en": "Why is the drink called bubble tea?",
        "answer_en": "Because shaking the tea makes foam on top.",
        "welcome_en": "Welcome to our listening channel.",
        "hook_intro_en": _hook(80),
        "outro": _hook(90),
        "youtube_tags": [f"tag{i}" for i in range(15)],
        "thumbnail_icons": [{"en": f"icon{i}", "zh": f"圖{i}"} for i in range(4)],
        "scene_images": [{"prompt": f"scene {i}", "label": f"s{i}"} for i in range(8)],
        "key_words": [],
        "dialogue": dlg,
    }


def _defect_script():
    """注入各 category 缺陷的脚本（最终 19 行，要求 20）。"""
    s = _clean_script()
    dlg = s["dialogue"]
    # 先做定点变异（基于 clean 的 20 行索引）
    dlg[1]["zh"] = "当然。它到底是什么？"                      # 简体字
    dlg[3]["zh"] = ""                                           # 空 zh
    dlg[6]["on_screen"] = ["char_c"]                            # 失去一个环境镜头
    dlg[8]["on_screen"] = ["char_c", "char_a"]                  # 顺序不规范
    dlg[12]["text"] = "Because shaking the tea makes foam, you know."  # 答案泄露
    dlg[13]["text"] = dlg[2]["text"]                            # 完全重复行
    dlg[17]["text"] = "Yes, it was a fun day for sure."         # reveal 丢失答案
    # 再删 3 行（review×2 + reveal×1）→ 17 行
    del dlg[19]
    del dlg[18]
    del dlg[16]
    # 插入 2 行 → 19 行（总行数 19 != 20）
    dlg.insert(4, _line("char_c", "Hello my friend, welcome.", "buildup",
                        "你好我的朋友，歡迎光臨。", ["char_c"]))  # char_c 泄漏到 buildup
    dlg.insert(10, _line("char_b", "Goodbye, see you tomorrow.", "review",
                         "再見，明天見。", ["char_a", "char_b"]))  # 阶段回退
    # 元数据
    s["hook_intro_en"] = "Too short hook."
    s["youtube_tags"] = ["a", "b"]
    s["_requested_num_lines"] = 20
    return s


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class TestCleanFixture(unittest.TestCase):
    def test_clean_passes(self):
        report = run_quality_gate(_clean_script(), num_lines=20)
        errors = [i for i in report["issues"] if i["severity"] == "error"]
        self.assertEqual(errors, [], f"clean fixture should have 0 errors:\n{format_report(report)}")
        self.assertTrue(report["passed"])


class TestDefectFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_quality_gate(_defect_script(), num_lines=20)

    def _issues(self, check, severity):
        return [i for i in self.report["issues"]
                if i["check"] == check and i["severity"] == severity]

    def test_line_count_error(self):
        errs = self._issues("structure", "error")
        self.assertTrue(any("总行数" in i["detail"] for i in errs))

    def test_phase_count_error(self):
        errs = self._issues("structure", "error")
        self.assertTrue(any("buildup 行数" in i["detail"] for i in errs))

    def test_phase_regression_error(self):
        errs = self._issues("structure", "error")
        self.assertTrue(any("阶段顺序回退" in i["detail"] for i in errs))

    def test_char_c_in_buildup_error(self):
        errs = self._issues("structure", "error")
        self.assertTrue(any("buildup) speaker 非法" in i["detail"] for i in errs))

    def test_empty_zh_error(self):
        errs = self._issues("fields", "error")
        self.assertTrue(any("'zh' 为空" in i["detail"] for i in errs))

    def test_duplicate_error(self):
        errs = self._issues("duplicates", "error")
        self.assertTrue(any("完全重复行" in i["detail"] for i in errs))

    def test_answer_leak_error(self):
        errs = self._issues("story", "error")
        self.assertTrue(any("泄露" in i["detail"] for i in errs))

    def test_reveal_missing_answer_error(self):
        errs = self._issues("story", "error")
        self.assertTrue(any("未明确揭晓答案" in i["detail"] for i in errs))

    def test_hook_too_short_warning(self):
        warns = self._issues("metadata", "warning")
        self.assertTrue(any("hook_intro_en" in i["detail"] for i in warns))

    def test_on_screen_order_warning(self):
        warns = self._issues("fields", "warning")
        self.assertTrue(any("顺序" in i["detail"] for i in warns))

    def test_simplified_zh_warning(self):
        warns = self._issues("translation", "warning")
        self.assertTrue(any("简体" in i["detail"] for i in warns))

    def test_env_shots_warning(self):
        # 缺陷夹具环境镜头 1 个（clean 有 2 个，menu 行被改为 ["char_c"]），低于下限 2
        warns = self._issues("fields", "warning")
        self.assertTrue(any("环境镜头" in i["detail"] for i in warns))

    def test_report_shape(self):
        self.assertIn("passed", self.report)
        self.assertIn("summary", self.report)
        self.assertFalse(self.report["passed"])


_REAL_SCRIPT = Path(
    r"H:\2026_main_project\colab_listening_b\output\【英文聽力挑戰】市場講價｜❓你能聽出答案嗎？｜A2慢速英文｜不用背多聽就會用｜英文聽力訓練｜Bargaining_at_the_Market\script.json")


@unittest.skipUnless(_REAL_SCRIPT.exists(), "真实生产脚本不存在，跳过集成回归")
class TestRealProductionScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = json.loads(_REAL_SCRIPT.read_text(encoding="utf-8"))
        cls.report = run_quality_gate(cls.script, num_lines=250)

    def test_known_errors_detected(self):
        details = " | ".join(i["detail"] for i in self.report["issues"]
                             if i["severity"] == "error")
        self.assertIn("总行数 266", details)
        self.assertIn("buildup 行数 85", details)
        self.assertIn("core 行数 126", details)
        self.assertIn("完全重复行", details)

    def test_known_warnings_detected(self):
        details = " | ".join(i["detail"] for i in self.report["issues"]
                             if i["severity"] == "warning")
        self.assertIn("环境镜头", details)
        self.assertIn("hook_intro_en", details)
        self.assertIn("outro", details)
        self.assertIn("youtube_tags", details)
        self.assertIn("thumbnail_icons", details)
        self.assertIn("answer_en", details)  # 旧管线组装脚本缺该字段

    def test_fails_overall(self):
        self.assertFalse(self.report["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
