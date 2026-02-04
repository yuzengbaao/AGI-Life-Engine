#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAD操作学习系统
实时观察、记录、理解用户的CAD绘图操作

功能概述：
- 记录操作（时间戳、类型、细节、上下文快照）
- 推断绘图类型、检测模式、评估技能水平、预测下步操作
- 生成可复用技能模板，并持久化到本地JSON
- 提供简易CLI演示入口
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


class CADLearningSystem:
    """CAD操作学习与记录系统"""

    def __init__(self, save_dir: Optional[str] = None):
        self.observation_log: List[Dict[str, Any]] = []
        self.operation_sequence: List[str] = []
        self.skill_library: Dict[str, Dict[str, Any]] = {}
        self.current_context: Dict[str, Any] = {
            "drawing_type": None,
            "design_intent": None,
            "current_layer": None,
            "active_tools": [],
        }
        self.session: Optional[Dict[str, Any]] = None

        # 持久化路径
        base_dir = (
            save_dir
            or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "memory")
        )
        os.makedirs(base_dir, exist_ok=True)
        self.skills_path = os.path.join(base_dir, "cad_learned_skills.json")
        self.sessions_path = os.path.join(base_dir, "cad_observation_sessions.jsonl")

        # 预加载技能库
        self._load_skills()

    # --------------------------- 会话与记录 ---------------------------
    def start_observation(self) -> str:
        """开始观察会话"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = {
            "id": session_id,
            "start_time": datetime.now().isoformat(),
            "observations": [],
        }
        return session_id

    def end_observation(self) -> None:
        """结束会话并保存会话日志"""
        if not self.session:
            return
        self.session["end_time"] = datetime.now().isoformat()
        # 写入JSONL，便于后续分析
        with open(self.sessions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.session, ensure_ascii=False) + "\n")
        self.session = None

    def record_operation(self, operation_type: str, details: Dict[str, Any]) -> None:
        """记录单个CAD操作"""
        observation = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "details": details,
            "context_snapshot": self.current_context.copy(),
        }

        self.observation_log.append(observation)
        self.operation_sequence.append(operation_type)

        # 更新上下文
        self._update_context(operation_type, details)

        # 写入当前会话
        if self.session is not None:
            self.session["observations"].append(observation)

    def _update_context(self, operation_type: str, details: Dict[str, Any]) -> None:
        """根据操作更新上下文理解"""
        if operation_type == "layer_change":
            self.current_context["current_layer"] = details.get("layer_name")
        elif operation_type == "drawing_start":
            self.current_context["drawing_type"] = details.get("drawing_type")
            self.current_context["design_intent"] = details.get("design_intent")
        elif operation_type == "tool_selection":
            tool = details.get("tool_name")
            if tool and tool not in self.current_context["active_tools"]:
                self.current_context["active_tools"].append(tool)

    # --------------------------- 推断与预测 ---------------------------
    def infer_intent(self) -> Dict[str, Any]:
        """从操作序列推断设计意图"""
        if not self.operation_sequence:
            return {"status": "no_operations", "message": "尚未记录操作"}

        intent_analysis = {
            "probable_drawing_type": self._infer_drawing_type(),
            "detected_patterns": self._detect_patterns(),
            "skill_level_indicator": self._assess_skill_level(),
            "next_operation_prediction": self._predict_next_operation(),
        }

        return intent_analysis

    def _infer_drawing_type(self) -> str:
        """推断绘图类型"""
        ops = set(self.operation_sequence)
        if {"line", "circle"}.issubset(ops):
            return "mechanical_drawing"
        elif {"polyline", "hatch"}.issubset(ops):
            return "architectural_plan"
        elif {"dimension", "text"}.issubset(ops):
            return "technical_drawing"
        else:
            return "general_drafting"

    def _detect_patterns(self) -> List[str]:
        """检测操作模式"""
        patterns: List[str] = []
        ops = set(self.operation_sequence)

        if {"line", "offset"}.issubset(ops):
            patterns.append("轮廓绘制与偏移")
        if {"circle", "array"}.issubset(ops):
            patterns.append("圆形阵列模式")
        if {"layer_change", "color_change"}.issubset(ops):
            patterns.append("图层与颜色管理")

        return patterns

    def _assess_skill_level(self) -> str:
        """评估用户技能水平（启发式）"""
        unique_operations = len(set(self.operation_sequence))
        if unique_operations < 3:
            return "beginner"
        elif unique_operations < 6:
            return "intermediate"
        else:
            return "advanced"

    def _predict_next_operation(self) -> str:
        """预测下一个可能操作（简单规则）"""
        if not self.operation_sequence:
            return "drawing_start"
        last_op = self.operation_sequence[-1]
        predictions = {
            "line": "dimension",
            "circle": "array",
            "rectangle": "offset",
            "dimension": "text",
            "hatch": "layer_change",
        }
        return predictions.get(last_op, "continue_drawing")

    # --------------------------- 技能模板 ---------------------------
    def generate_skill_template(self) -> Dict[str, Any]:
        """生成可复用的技能模板并写入技能库"""
        if len(self.observation_log) < 2:
            return {"status": "insufficient_data", "message": "观测不足，至少需要2步"}

        skill_id = f"cad_skill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        template = {
            "skill_id": skill_id,
            "name": self._generate_skill_name(),
            "description": self._generate_skill_description(),
            "drawing_type": self.current_context.get("drawing_type"),
            "prerequisites": self._identify_prerequisites(),
            "operation_sequence": self.operation_sequence.copy(),
            "detailed_steps": self._extract_detailed_steps(),
            "parameters": self._extract_parameters(),
            "common_variations": self._suggest_variations(),
            "generated_at": datetime.now().isoformat(),
        }

        # 保存到技能库（内存+文件）
        self.skill_library[skill_id] = template
        self._save_skills()
        return template

    def _generate_skill_name(self) -> str:
        drawing_type = self.current_context.get("drawing_type") or "drawing"
        main_ops = list(dict.fromkeys(self.operation_sequence[:3]))  # 去重保序
        op_names = "_".join(main_ops[:2]) if main_ops else "basic"
        return f"{drawing_type}_{op_names}_skill"

    def _generate_skill_description(self) -> str:
        intent = self.current_context.get("design_intent") or "绘图"
        ops = len(set(self.operation_sequence))
        return f"实现{intent}的CAD操作序列，包含{ops}种不同操作"

    def _extract_detailed_steps(self) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for i, obs in enumerate(self.observation_log, start=1):
            steps.append(
                {
                    "step": i,
                    "operation_type": obs.get("operation_type"),
                    "details": obs.get("details", {}),
                    "context": obs.get("context_snapshot", {}),
                }
            )
        return steps

    def _extract_parameters(self) -> Dict[str, Any]:
        """从细节中提取可能的参数集合（启发式）"""
        params: Dict[str, Any] = {}
        for obs in self.observation_log:
            details: Dict[str, Any] = obs.get("details", {})
            for k, v in details.items():
                # 只保留基础类型参数
                if isinstance(v, (str, int, float)):
                    # 记录最后一次出现的值
                    params[k] = v
        return params

    def _identify_prerequisites(self) -> List[str]:
        """识别前置条件（图层/工具等）"""
        prereq: List[str] = []
        if self.current_context.get("current_layer"):
            prereq.append(f"layer:{self.current_context['current_layer']}")
        for tool in self.current_context.get("active_tools", []):
            prereq.append(f"tool:{tool}")
        return prereq

    def _suggest_variations(self) -> List[str]:
        """基于常见操作提供变体建议"""
        variations: List[str] = []
        ops = set(self.operation_sequence)
        if "offset" in ops:
            variations.append("可替换为scale进行比例缩放")
        if "array" in ops:
            variations.append("可替换为copy实现手动阵列")
        if "hatch" in ops:
            variations.append("可更换填充图案或角度")
        return variations

    # --------------------------- 持久化 ---------------------------
    def _load_skills(self) -> None:
        if os.path.exists(self.skills_path):
            try:
                with open(self.skills_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.skill_library = data
            except Exception:
                # 若损坏则忽略，重新开始
                self.skill_library = {}

    def _save_skills(self) -> None:
        try:
            with open(self.skills_path, "w", encoding="utf-8") as f:
                json.dump(self.skill_library, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


# --------------------------- 演示入口 ---------------------------
if __name__ == "__main__":
    system = CADLearningSystem()
    sid = system.start_observation()
    print(f"[CAD学习系统] 开始观察会话 {sid}")
    print("准备记录您的CAD操作步骤...\n")

    # 模拟几步操作
    system.record_operation(
        "drawing_start",
        {"drawing_type": "architectural_plan", "design_intent": "房间平面图", "description": "开始建筑平面图绘制"},
    )
    system.record_operation("tool_selection", {"tool_name": "polyline", "description": "选择多段线工具"})
    system.record_operation("polyline", {"points": 5, "description": "绘制房间轮廓"})
    system.record_operation("hatch", {"pattern": "ANSI31", "description": "地面材质填充"})
    system.record_operation("dimension", {"style": "ISO", "description": "添加尺寸标注"})
    system.record_operation("text", {"content": "房间A", "description": "添加文字说明"})

    intent = system.infer_intent()
    print("\n[推断]", json.dumps(intent, ensure_ascii=False, indent=2))

    template = system.generate_skill_template()
    print("\n[生成技能模板]", json.dumps(template, ensure_ascii=False, indent=2))

    system.end_observation()
    print("\n[完成] 会话已保存，技能库已更新。")
