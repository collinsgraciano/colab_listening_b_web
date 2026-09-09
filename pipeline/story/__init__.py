"""Story mode package — 同款家庭故事/对话/独白三类型生成（用户可配置家庭角色）。

模块：
- llm_client_story: 三类型脚本生成器（plot/chat/solo）
- quality_gate_story: 程序化质检门禁
时间轴/SRT 复用 quest.timeline_quest（welcome/hook/outro 为空 → 纯对话时间轴）。
"""
