import time
import sys
import logging
import random
import os

# 🔧 [2026-01-29] 单实例保护 - 防止多进程运行
try:
    from core.single_instance_protection import ensure_single_instance
    SINGLE_INSTANCE_AVAILABLE = True
except ImportError:
    SINGLE_INSTANCE_AVAILABLE = False
    logging.warning("单实例保护模块不可用，可能导致多进程问题")

# 🔧 [2026-01-11] Fix Windows console encoding for emoji support
import io
if sys.platform == 'win32':
    # Reconfigure stdout and stderr to use UTF-8 encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Disable ChromaDB telemetry immediately to prevent PostHog errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

# Suppress PaddlePaddle/TensorFlow C++ logging noise (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress ChromaDB telemetry logger
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# 🔧 [2026-01-31] Audio Overflow Warning Fix
try:
    from core.perception.audio_overflow_fix import suppress_audio_overflow_warnings
    suppress_audio_overflow_warnings()
except ImportError:
    logging.warning("⚠️ Audio overflow fix module not found")

import datetime
import json
import re
import asyncio
import shlex
from typing import List, Dict, Any
from collections import deque

# 导入工具执行桥接层
try:
    from tool_execution_bridge import ToolExecutionBridge
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False

# 导入意图对话桥接层
try:
    from intent_dialogue_bridge import get_intent_bridge, IntentState, IntentDepth
    INTENT_BRIDGE_AVAILABLE = True
except ImportError:
    INTENT_BRIDGE_AVAILABLE = False

# --- Core Infrastructure Imports ---
from core.goal_system import GoalManager, GoalType, GoalStatus
from core.llm_client import LLMService
from core.system_tools import SystemTools
from core.desktop_automation import DesktopController
from core.vision_observer import VisionObserver
from core.macro_system import SkillLibrary, MacroPlayer
from core.knowledge_graph import ArchitectureKnowledgeGraph
from core.knowledge_reasoner import KnowledgeReasoner
from core.neuro_symbolic_bridge import NeuroSymbolicBridge
# from core.memory_enhanced import EnhancedExperienceMemory # Legacy Phase 5 Memory
from core.memory_enhanced_v2 import EnhancedExperienceMemory # Priority 1 Upgrade (LRU/Intuition)
from core.philosophy import MeaningOfExistenceExplorer
from core.layered_identity import ImmutableCore
from core.global_observer import GlobalObserver

# 🔧 [2026-01-15] 新增：导入Insight实用函数库（提升Insight可执行性）
try:
    from core.insight_utilities import (
        invert_causal_chain, perturb_attention_weights, simulate_forward,
        rest_phase_reorganization, noise_guided_rest, semantic_perturb,
        analyze_tone, semantic_diode, detect_topological_defect,
        fractal_idle_pulse, reverse_abduction_step, inject_adversarial_intuition,
        latent_recombination, kl_div, CurlLayer
    )
    INSIGHT_UTILITIES_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Insight实用函数库已加载 - 提升Insight可执行性")
except ImportError as e:
    INSIGHT_UTILITIES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Insight实用函数库不可用: {e}")
    logger.warning("   系统将继续运行，但Insight代码的可执行性可能受限")
from core.cad_observer import CADObserver
from core.intent_tracker import IntentTracker
# 🛠️ [FIX 2026-01-15] NumPy版本兼容性：进化控制器设为可选
try:
    from core.evolution.impl import EvolutionController
    from core.evolution.genesis import perform_genesis
    EVOLUTION_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"   [System] ⚠️ Evolution Controller不可用: {type(e).__name__}")
    print(f"   [System]    详细错误: {str(e)[:150]}")
    print(f"   [System] 🔄 系统将在无进化功能的情况下运行")
    EVOLUTION_AVAILABLE = False
    EvolutionController = None
    perform_genesis = None
from core.perception import PerceptionManager, WhisperASR, CaptureStatus
from core.perception.asr import StreamingWhisperASR, WhisperModelSize
from core.perception.monitor import extend_monitoring_with_perception, PerceptionMonitorExtension
from core.perception.runtime_monitor import RuntimeMonitor
from agi_component_coordinator import EventBus, Event, ComponentCoordinator
from security_framework import SecurityManager
from core.hardware_capture import HardwareCaptureManager, CameraConfig, MicrophoneConfig
from core.image_preprocessing import ImagePreprocessor, ColorSpace
from core.audio_preprocessing import AudioPreprocessor
from core.multimodal_fusion import MultimodalFusion, MultimodalDecisionSupport, ModalityData, ModalityType

# --- Core Agents Imports ---
from core.agents.planner import PlannerAgent
from core.agents.executor import ExecutorAgent
from core.agents.critic import CriticAgent
from core.foraging_agent import ForagingAgent  # 🆕 [2026-01-09] Active Learning Agent
from core.evolution.dynamics import EvolutionaryDynamics
# 🛠️ [FIX 2026-01-15] NumPy版本兼容性：神经记忆系统设为可选
try:
    from core.memory.neural_memory import BiologicalMemorySystem
    BIOLOGICAL_MEMORY_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"   [System] ⚠️ BiologicalMemorySystem不可用: {type(e).__name__}")
    print(f"   [System]    详细错误: {str(e)[:150]}")
    print(f"   [System] 🔄 系统将在无神经记忆功能的情况下运行")
    BIOLOGICAL_MEMORY_AVAILABLE = False
    BiologicalMemorySystem = None
from core.reasoning.arc_solver import ARCSolver # Program Synthesis Engine
from core.skill_manager import SkillManager
from core.motivation import MotivationCore  # 动力核心 (Maslow + Dopamine)
# 🆕 [2026-01-29] Real Perception System (Sentence Transformers)
from core.perception_system import PerceptionSystem

# 🆕 [2026-01-15] 双螺旋决策引擎V2 - 真正的智能融合
from core.double_helix_engine_v2 import DoubleHelixEngineV2

# Ensure log directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class LifeEngineEventBus:
    def __init__(self, source: str = "AGI_Life_Engine"):
        self._source = source
        self._bus = EventBus()

    def subscribe(self, event_type: str, handler):
        self._bus.subscribe(event_type, handler)

    async def publish(self, event_type: str, data: Dict[str, Any]):
        await self._bus.publish(Event(type=event_type, source=self._source, data=data))

class ExistentialLogger:
    """
    Handles the 'Existential Testimony' of the AGI.
    Generates audit, ethos, and sync logs as per the Self-Cognition definition.
    """
    def __init__(self):
        self.logger = logging.getLogger("ExistentialLogger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_audit(self, message: str, coherence_phi: float):
        phi_str = f"{int(coherence_phi * 100)}"
        filename = f"{LOG_DIR}/audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{phi_str}.log"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Coherence Phi: {coherence_phi}\n")
            f.write(f"Observation: {message}\n")
            f.write("Status: VERIFIED_SELF\n")

    def log_ethos(self, decision: str, hesitation_tau: float):
        tau_val = int(hesitation_tau * 100)
        filename = f"{LOG_DIR}/ethos_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_tau{tau_val}.log"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Hesitation Tau: {hesitation_tau}s\n")
            f.write(f"Decision: {decision}\n")
            f.write("Status: RESPONSIBILITY_ACCEPTED\n")

    def log_sync(self, residual: float, cycle: int):
        filename = f"{LOG_DIR}/sync_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_res{int(residual*1e8)}.log"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Residual: {residual}\n")
            f.write(f"Cycle: {cycle}\n")
            f.write("Status: DOUBT_ACTIVE\n")

    def log_cycle_flow(self, cycle_data: Dict):
        """
        Log the full cognitive cycle flow in structured JSON format.
        Implements the 'Structured Logging' suggestion for better analysis.
        """
        filename = f"{LOG_DIR}/flow_cycle.jsonl"
        try:
            # Handle non-serializable objects (like numpy arrays)
            import numpy as np
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.float64):
                        return float(obj)
                    if isinstance(obj, np.int64):
                        return int(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(cycle_data, cls=NumpyEncoder, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"   [Logger] ⚠️ Failed to log cycle flow: {e}")

def save_insight_markdown(insight_data: Dict[str, Any]) -> Dict[str, Any]:
    tmp_path = None
    try:
        ts_float = insight_data.get("timestamp", time.time())
        timestamp = int(ts_float)
        insight_dir = os.path.join("data", "insights")
        os.makedirs(insight_dir, exist_ok=True)

        filename = f"insight_{timestamp}.md"
        file_path = os.path.join(insight_dir, filename)

        content = insight_data.get("content", "") or ""
        entropy = insight_data.get("entropy_score", 0.0)
        try:
            entropy = float(entropy)
        except Exception:
            entropy = 0.0

        trigger_goal = insight_data.get("trigger_goal", "Unknown")
        bridge_validation = insight_data.get("bridge_validation", {})
        try:
            bridge_validation_str = json.dumps(bridge_validation, ensure_ascii=False)
        except Exception:
            bridge_validation_str = str(bridge_validation)

        def normalize_sections(raw: str) -> str:
            raw = raw or ""
            has_hypothesis = re.search(r"(?im)^Hypothesis\\s*:", raw) is not None
            has_insight = re.search(r"(?im)^Insight\\s*:", raw) is not None
            code_block_match = re.search(r"```python[\\s\\S]*?```", raw)

            if has_hypothesis and has_insight:
                if code_block_match and re.search(r"(?im)^(Code Snippet|Code)\\s*:", raw) is None:
                    insert_at = code_block_match.start()
                    raw = f"{raw[:insert_at].rstrip()}\\n\\nCode Snippet:\\n{raw[insert_at:].lstrip()}"
                return raw.strip()

            code_block = ""
            remaining = raw.strip()
            if code_block_match:
                code_block = code_block_match.group(0).strip()
                remaining = (raw[:code_block_match.start()].strip() + "\n\n" + raw[code_block_match.end():].strip()).strip()

            parts = []
            if remaining:
                parts.append(f"Insight:\n{remaining}\n")
            else:
                parts.append("Insight:\n\n")
            parts.append("Hypothesis:\n\n")
            parts.append("Code Snippet:\n")
            if code_block:
                parts.append(f"{code_block}\n")
            return "\n".join(parts).strip()

        normalized_content = normalize_sections(content)

        markdown = (
            f"# Creative Insight (Entropy: {entropy})\n\n"
            f"Trigger Goal: {trigger_goal}\n"
            f"Validation: {bridge_validation_str}\n\n"
            f"{normalized_content}\n"
        )

        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(markdown)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)

        return {
            "success": True,
            "file_path": file_path,
            "abs_path": os.path.abspath(file_path),
            "timestamp": timestamp
        }
    except Exception as e:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return {"success": False, "error": str(e)}

class SystemMonitor:
    """
    Central monitoring layer for the AGI system.
    """
    def __init__(self):
        self.logger = logging.getLogger("SystemMonitor")
        self.perception_monitor: PerceptionMonitorExtension = None

    def capture_exception(self, error: Exception, context: Dict = None, severity: str = 'error', component: str = 'unknown'):
        msg = f"[{component}] {severity.upper()}: {error} | Context: {context}"
        if severity == 'error':
            self.logger.error(msg)
        else:
            self.logger.warning(msg)

import threading
import queue

class ConsoleInputListener:
    """
    Asynchronous Console Input Listener.
    Runs in a separate thread to avoid blocking the main AGI loop.
    Captures user commands from stdin and pushes them to a queue.
    """
    def __init__(self):
        self.command_queue = queue.Queue()
        self.is_running = False
        self.listener_thread = None

    def start(self):
        self.is_running = True
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        print("   [System] ⌨️  Console Input Listener Online. Type 'help' for commands.")

    def stop(self):
        self.is_running = False
        # Thread is daemon, will die with main process, but we can try to be clean
        # (input() is blocking, so hard to kill gracefully without OS signals)

    def _listen_loop(self):
        print("   [Input] Listener ready. You can type commands directly here (ignore log scrolling):")
        while self.is_running:
            try:
                # This blocks, but it's in a thread so main loop is fine
                user_input = input() 
                if user_input.strip():
                    self.command_queue.put(user_input.strip())
            except EOFError:
                break
            except Exception as e:
                print(f"   [Input] Error: {e}")
                break

    def get_command(self):
        if not self.command_queue.empty():
            return self.command_queue.get_nowait()
        return None

class AGI_Life_Engine:
    """
    The Core Life Engine of the AGI System.
    NOW RECONNECTED TO THE PHYSICAL BODY (Core Infrastructure).
    Includes Soul Evolution (Philosopher) and Constitutional Alignment (Immutable Core).
    """
    def _cleanup_startup_cache(self):
        base_dir = os.getcwd()
        targets = [
            os.path.join(base_dir, "data", "intent_bridge", "user_intents.jsonl"),
            os.path.join(base_dir, "data", "intent_bridge", "engine_responses.jsonl"),
            os.path.join(base_dir, "data", "intent_bridge", "active_intent.json"),
            os.path.join(base_dir, "data", "next_tasks.json"),
            # [FIXED 2026-01-29] Removed: os.path.join(base_dir, "logs", "flow_cycle.jsonl"),
            # [FIX 2026-01-27] 拓扑图已修复，不应在启动时删除
            # os.path.join(base_dir, "data", "neural_memory", "topology_graph.json"),
            os.path.join(base_dir, "data", "neural_memory", "topology_visual.json"),
            os.path.join(base_dir, "data", "neural_memory", "topology_graph.mmd")
        ]
        removed = []
        for path in targets:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed.append(os.path.relpath(path, base_dir))
                except Exception as e:
                    print(f"   [System] ⚠️ Cache cleanup failed: {path} ({e})")
        if removed:
            print("   [System] 🧹 Startup cache cleaned:")
            for item in removed:
                print(f"      - {item}")
        else:
            print("   [System] 🧹 Startup cache cleaned: none")

    def __init__(self):
        self.is_running = True
        self.step_count = 0
        self.context = {"status": "active", "mode": "learning"}

        # 🆕 [ADAPTIVE POLLING 2026-01-27] 自适应轮询器
        self._adaptive_poller = None  # 在intent_bridge初始化后创建
        self._last_poll_tick = 0  # 上次轮询的TICK计数
        self.existential_logger = ExistentialLogger()
        self.start_time = time.time()
        self._cleanup_startup_cache()
        self.last_goal_id = None
        self.current_plan = []          # 状态化：当前执行计划
        self.current_step_index = 0     # 状态化：当前步骤索引
        self.failed_steps_for_current_goal = []
        self.last_evolution_guidance = None
        self.last_insight_creation_ts = 0.0
        self._insight_persist_failure_count = 0
        self._insight_persist_failure_ts = 0.0
        self._insight_persist_backoff_until = 0.0
        
        # 🔧 [2026-01-11] 元认知调查冷却机制 - 防止空转循环
        self._last_meta_investigation_ts = 0.0
        self._meta_investigation_cooldown = 300  # 5分钟冷却期
        self._curiosity_satisfaction_decay = 0.0  # 好奇心满足衰减
        
        print("   [System] 🧬 Initializing Organic Architecture (Learning Mode)...")
        
        # 0. Load Immutable Core (Constitution)
        self.core_identity = ImmutableCore()
        print(f"   [System] 📜 Constitution Loaded: {self.core_identity.system_name} {self.core_identity.version}")
        
        # 0.5 Initialize System Monitor
        self.monitor = SystemMonitor()
        RuntimeMonitor.register(self.monitor, context_info="System Monitor")

        # 0.55 Initialize Event Bus
        self.event_bus = LifeEngineEventBus(source="AGI_Life_Engine")
        RuntimeMonitor.register(self.event_bus, context_info="Event Bus")

        # Subscribe to Insight Generation for Persistence
        self.event_bus.subscribe("insight_generated", self._on_insight_generated)

        # 0.6 Initialize Console Input
        self.console_listener = ConsoleInputListener()
        self.console_listener.start()

        # 🆕 [2026-01-29] Initialize Real Perception System (The Eye)
        # Replaces MD5 hashing with SentenceTransformers
        self.perception_system = PerceptionSystem()

        # 1. Initialize Brain (LLM)
        self.llm_service = LLMService()
        if self.llm_service.mock_mode:
            print("   [System] ⚠️ Warning: Running in MOCK mode (No API Key found).")
        else:
            print(f"   [System] 🧠 Connected to {self.llm_service.active_provider}")

        # 2. Initialize Body (Tools & Senses)
        self.system_tools = SystemTools(work_dir=os.getcwd())
        self.desktop = DesktopController()
        
        # Initialize Macro System (Skill Memory & Playback)
        self.skill_library = SkillLibrary()
        self.macro_player = MacroPlayer(self.desktop, self.skill_library)
        print("   [System] 🦾 Macro Automation System Online.")

        self.vision = VisionObserver()
        self.global_observer = GlobalObserver()
        self.cad_observer = CADObserver()
        self.intent_tracker = IntentTracker()
        self.memory = ArchitectureKnowledgeGraph()

        # 🆕 [2026-01-30] P1修复: 初始化孤立节点预防器
        self.isolation_prevention = None
        try:
            from core.isolated_node_prevention import create_isolation_prevention
            self.isolation_prevention = create_isolation_prevention(self.memory)
            print("   [System] 🔗 Isolated Node Prevention Online")
        except Exception as e:
            print(f"   [System] ⚠️ Isolated Node Prevention initialization failed: {e}")

        # 🛠️ [FIX 2026-01-15] 添加异常处理：EnhancedMemory可能因ChromaDB问题失败
        try:
            self.semantic_memory = EnhancedExperienceMemory()
            print("   [System] ✅ Enhanced Experience Memory V2 Online")
        except Exception as e:
            print(f"   [System] ⚠️ Enhanced Memory初始化失败: {type(e).__name__}")
            print(f"   [System]    错误详情: {str(e)[:100]}")
            print("   [System] 🔄 降级到SimpleMemory系统 (保证系统稳定运行)")
            try:
                from core.memory_simple import SimpleMemorySystem
                self.semantic_memory = SimpleMemorySystem()
                print("   [System] ✅ Simple Memory Online (Fallback Mode)")
            except Exception as e2:
                print(f"   [System] ❌ SimpleMemory也失败了: {e2}")
                print("   [System] 🔄 使用空字典作为最后防线")
                self.semantic_memory = {}  # 空字典，系统仍能运行
        
        # 🆕 Biological Memory (Fluid Intelligence)
        if BIOLOGICAL_MEMORY_AVAILABLE:
            self.biological_memory = BiologicalMemorySystem()
            print(f"   [System] 🧠 Biological Memory Online ({self.biological_memory.topology.size()} nodes)")
            self.system_tools.biological_memory = self.biological_memory
        else:
            self.biological_memory = None
            print("   [System] ⏸️  Biological Memory 不可用 (系统将以简化模式运行)")
        
        # ✅ [FIX 2026-01-09] 初始化TopologyMemory并建立记忆系统桥接
        from core.memory.topology_memory import TopologicalMemoryCore
        self.topology_memory = TopologicalMemoryCore()
        print(f"   [System] 🕸️ Topology Memory Online")
        
        # 建立记忆系统之间的连接（Layer3拓扑修复）
        # BiologicalMemory ↔ TopologyMemory ↔ KnowledgeGraph
        # 注意：实际桥接逻辑需要在各模块内部实现，这里只是建立引用关系
        self.biological_memory._topology_ref = self.topology_memory  # 存储拓扑引用
        self.memory._topology_ref = self.topology_memory  # KnowledgeGraph也连接拓扑
        
        self.reasoner = KnowledgeReasoner(self.memory)
        self.arc_solver = ARCSolver() # Initialize Program Synthesis Engine
        print("   [System] 🧩 ARC Solver (Program Synthesis) Online.")
        
        # Initialize Neuro-Symbolic Bridge (The Connector)
        self.neuro_bridge = NeuroSymbolicBridge()
        print("   [System] 🧠 NeuroSymbolic Bridge (Semantic Drift Detection) Online.")
        
        # Hydrate Bridge with existing knowledge to prevent "Amnesia"
        # We sync the topological structure so 'surprise' metrics are valid
        if self.memory.graph.number_of_nodes() > 0:
            print(f"   [Bridge] Hydrating from Knowledge Graph ({self.memory.graph.number_of_nodes()} nodes)...")
            self.neuro_bridge.update_topology(
                nodes=list(self.memory.graph.nodes()),
                edges=list(self.memory.graph.edges())
            )
        
        RuntimeMonitor.register(self.memory, context_info="Long-term Memory (Knowledge Graph)")
        RuntimeMonitor.register(self.semantic_memory, context_info="Semantic Memory V2 (ChromaDB + Intuition)")
        RuntimeMonitor.register(self.biological_memory, context_info="Biological Memory (Fluid Topology)")
        RuntimeMonitor.register(self.reasoner, context_info="Reasoning Engine")
        RuntimeMonitor.register(self.neuro_bridge, context_info="Neuro-Symbolic Bridge")
        
        # 2.5 Initialize Extended Perception (Hearing & Real-time Vision)
        try:
            self.perception = PerceptionManager()
            self.perception.start_all()
            
            # Initialize ASR (Use TINY for speed if needed, but BASE is standard)
            self.whisper = WhisperASR(model_size=WhisperModelSize.BASE)
            self.streaming_asr = StreamingWhisperASR(self.whisper)
            self.streaming_asr.start()
            
            # Connect Perception -> ASR
            def audio_callback(data):
                if 'audio' in data:
                    audio = data['audio']
                    if hasattr(audio, 'ndim') and audio.ndim > 1:
                        audio = audio.flatten()
                    self.streaming_asr.add_audio(audio)
            
            self.perception.set_audio_processor(audio_callback)
            
            # Attach Monitoring
            extend_monitoring_with_perception(self.monitor, self.perception)
            
            print("   [System] 👂 Hearing (Whisper) & 👁️ Real-time Vision Online.")
        except Exception as e:
            print(f"   [System] ⚠️ Extended Perception Init Failed: {e}")
            self.perception = None
            self.streaming_asr = None

        # 2.6 Initialize Hardware Capture (Camera & Microphone)
        try:
            self.hardware_capture = HardwareCaptureManager(
                camera_config=CameraConfig(camera_id=0, width=640, height=480, fps=30),
                microphone_config=MicrophoneConfig(sample_rate=16000, channels=1),
                enable_camera=True,
                enable_microphone=True
            )
            if self.hardware_capture.start_all():
                print("   [System] 📷 Camera & 🎤 Microphone Hardware Capture Online.")
            else:
                print("   [System] ⚠️ Hardware Capture Init Failed")
                self.hardware_capture = None
        except Exception as e:
            print(f"   [System] ⚠️ Hardware Capture Init Failed: {e}")
            self.hardware_capture = None

        # 2.7 Initialize Preprocessing & Fusion
        try:
            self.image_preprocessor = ImagePreprocessor(target_size=(224, 224))
            self.audio_preprocessor = AudioPreprocessor(sample_rate=16000)
            self.multimodal_fusion = MultimodalFusion(visual_weight=0.6, audio_weight=0.4)
            self.multimodal_decision = MultimodalDecisionSupport(self.multimodal_fusion)
            print("   [System] 🎨 Image & 🎵 Audio Preprocessing & Fusion Online.")
        except Exception as e:
            print(f"   [System] ⚠️ Preprocessing & Fusion Init Failed: {e}")
            self.image_preprocessor = None
            self.audio_preprocessor = None
            self.multimodal_fusion = None
            self.multimodal_decision = None

        print("   [System] 👁️ Vision & 🖐️ Manipulation Systems Online.")
        print("   [System] 🌍 Global Awareness & Intent Tracking Online.")
        print("   [System] 📐 Rule-Based Logic Engine (Recursive Flow) Online.")

        # 3. Initialize Goal System
        self.goal_manager = GoalManager(base_path=os.getcwd())
        self.recent_goals = deque(maxlen=5)

        # 4. Initialize Agents (The Trinity)
        # 🆕 [2026-01-18] Planner 现在接收 event_bus 以感知动态创建的工具
        self.planner = PlannerAgent(
            self.llm_service, 
            biological_memory=self.biological_memory,
            event_bus=self.event_bus,
            tool_registry=getattr(self, 'tool_factory', None)  # 若 tool_factory 已初始化
        )
        self.executor = ExecutorAgent(self.llm_service, self.system_tools, self.desktop)
        self.executor.biological_memory = self.biological_memory
        self.executor.macro_player = self.macro_player
        self.critic = CriticAgent(self.llm_service)
        
        # Register Agents for Runtime Monitoring
        RuntimeMonitor.register(self.planner, context_info="Planner Agent (Trinity)")
        RuntimeMonitor.register(self.executor, context_info="Executor Agent (Trinity)")
        RuntimeMonitor.register(self.critic, context_info="Critic Agent (Trinity)")
        
        print("   [System] 🤖 Agents (Planner, Executor, Critic) Active.")

        # 5. Initialize Philosopher (Soul Evolution)
        # 🆕 [2026-01-28] Inject LLM for REAL recursive consciousness
        self.meaning_explorer = MeaningOfExistenceExplorer(self.llm_service)
        print("   [System] 🧠 Philosopher Component (Recursive Consciousness) Online.")

        # 6. Initialize Evolution Controller (The New Essence)
        if EVOLUTION_AVAILABLE:
            self.evolution_controller = EvolutionController(self.llm_service)
            RuntimeMonitor.register(self.evolution_controller, context_info="Evolution Controller (The Seed)")
            print("   [System] 🧬 Evolution Controller (Self-Modification & World Model) Online.")
        else:
            self.evolution_controller = None
            print("   [System] ⏸️  Evolution Controller 不可用 (系统将以简化模式运行)")

        # 8. Initialize Skill Manager (Dynamic Capability)
        self.skill_manager = SkillManager()
        print("   [System] 🛠️ Skill Manager (Dynamic Capability) Online.")
        
        # 🆕 [2026-01-09] Initialize Insight Validation-Integration-Evaluation Loop
        # 🔧 [2026-01-10] 升级为增强验证器（解决伪代码问题）
        from core.insight_validator import InsightValidator
        from core.insight_integrator import InsightIntegrator
        from core.insight_evaluator import InsightEvaluator
        
        # 构建系统依赖图（记录AGI系统中已存在的函数）
        system_dependency_graph = self._build_system_dependency_graph()
        
        self.insight_validator = InsightValidator(
            system_dependency_graph=system_dependency_graph
        )
        self.insight_integrator = InsightIntegrator()
        self.insight_evaluator = InsightEvaluator()
        print(f"   [System] 🔬 Insight V-I-E Loop (Enhanced Validation) Online. Registered {len(system_dependency_graph)} system functions.")
        
        # 🆕 [2026-01-09] Initialize Foraging Agent (Active Learning)
        self.foraging_agent = ForagingAgent(curiosity_threshold=0.7, exploration_budget=10)
        print("   [System] 🔍 Foraging Agent (Active Learning) Online.")
        
        # 8.5 🆕 Initialize Motivation Core (The Drive) - "身心合一"
        self.motivation = MotivationCore()
        print("   [System] 🔥 Motivation Core (Maslow + Dopamine) Online.")
        
        # 🆕 [2026-01-15] Initialize Double Helix Engine V2 (真正的智能决策引擎)
        # 这是新旧系统真正融合的核心 - 将双螺旋决策引擎集成到完整AGI基础设施
        try:
            self.helix_engine = DoubleHelixEngineV2(
                state_dim=64,
                action_dim=8,  # 扩展动作空间：基础4 + 创造性4
                device='cpu',
                enable_nonlinear=True,
                enable_meta_learning=True,
                enable_dialogue=False
            )
            self.helix_decision_enabled = True
            print("   [System] 🧬 Double Helix Engine V2 Online - Dual-System Decision Making Enabled.")
            print("      [Helix] ⚡ System A (TheSeed) + System B (FractalIntelligence) Fusion Active")
        except Exception as e:
            self.helix_engine = None
            self.helix_decision_enabled = False
            print(f"   [System] ⚠️ Double Helix Engine V2 initialization failed: {e}")
        
        # 🆕 [2026-01-10] Initialize Security Manager (System-Level)
        # 提升为系统级组件，而非仅在Bridge内部使用
        self.security_manager = SecurityManager()
        print("   [System] 🛡️ Security Manager (System-Level) Online.")
        
        # 🆕 [2026-01-10] Initialize ImmutableCore Bridge (核心策略桥接)
        # 实现拓扑连接: ImmutableCore → SecurityManager, ImmutableCore → CriticAgent
        try:
            from core.immutable_core_bridge import ImmutableCoreBridge
            self.immutable_core_bridge = ImmutableCoreBridge()
            # 将核心策略注入到 SecurityManager（如果支持）
            if hasattr(self.security_manager, 'set_policy_source'):
                self.security_manager.set_policy_source(self.immutable_core_bridge)
            print("   [System] 🧬 ImmutableCore Bridge Online - Core directives connected.")
        except ImportError as e:
            self.immutable_core_bridge = None
            print(f"   [System] ⚠️ ImmutableCore Bridge not available: {e}")
        
        # 🆕 [2026-01-10] Initialize Component Coordinator (EventBus Hub)
        # 修复拓扑图中ComponentCoordinator未接入的问题
        self.component_coordinator = ComponentCoordinator(agi_system=self)
        # 让SecurityManager通过Coordinator可访问
        self.component_coordinator.register_component("security", self.security_manager)
        # 注册 ImmutableCore Bridge
        if self.immutable_core_bridge:
            self.component_coordinator.register_component("core_policy", self.immutable_core_bridge)
        print(f"   [System] 📌 Component Coordinator Online - EventBus Hub enabled.")
        
        # 9. Initialize Tool Execution Bridge (LLM→Real Execution)
        self.tool_bridge = None
        self._capability_prompt = ""  # LLM注入的工具能力提示词
        self._introspection_mode = True  # 🔧 [2026-01-30] 启用内省自修复模式
        if BRIDGE_AVAILABLE:
            self.tool_bridge = ToolExecutionBridge(agi_system=self)
            # 🆕 [2026-01-30] 内省模式：使用自修复能力提示词
            try:
                from core.introspection_mode import INTROSPECTION_CAPABILITY_PROMPT
                self._capability_prompt = INTROSPECTION_CAPABILITY_PROMPT
                print(f"   [System] 🔍 Introspection Mode ENABLED - Focused on self-repair")
            except ImportError:
                # 回退到标准工具提示
                if hasattr(self.tool_bridge, 'get_capability_prompt'):
                    self._capability_prompt = self.tool_bridge.get_capability_prompt()
            # 打印已文档化工具数
            caps = self.tool_bridge.get_tool_capabilities() if hasattr(self.tool_bridge, 'get_tool_capabilities') else {}
            doc_count = caps.get('documented_tools', 0) if isinstance(caps, dict) else 0
            print(f"   [System] 🔧 Tool Execution Bridge Online - {doc_count} tools documented, LLM capability injection ready.")
        else:
            print("   [System] ⚠️ Tool Execution Bridge not available.")
        
        # 10. Initialize Intent Dialogue Bridge (Bidirectional Intent Communication)
        self.intent_bridge = None
        if INTENT_BRIDGE_AVAILABLE:
            # 🆕 [ENHANCE 2026-01-27] 传递cognitive_bridge（初始化时可能为None，后续更新）
            # 🆕 [ENHANCE 2026-01-27] 启用Redis IPC
            # 注意：此时 self.cognitive_bridge 还未初始化，后续会更新
            self.intent_bridge = get_intent_bridge(
                llm_service=self.llm_service,
                cognitive_bridge=None,  # 初始为None，后续在CognitiveBridge初始化后更新
                use_redis=True,  # 🆕 启用Redis IPC
                redis_host='localhost',
                redis_port=6379
            )
            print("   [System] 🔗 Intent Dialogue Bridge Online - Deep intent understanding enabled.")
            print("   [System] 🚀 Redis IPC enabled for high-performance messaging.")
        else:
            print("   [System] ⚠️ Intent Dialogue Bridge not available.")

        # 🔧 [2026-01-15] P0修复: 激活BridgeAutoRepair自修复功能
        self.bridge_auto_repair = None
        try:
            from bridge_auto_repair import BridgeAutoRepair
            self.bridge_auto_repair = BridgeAutoRepair(
                bridge_file_path="tool_execution_bridge.py",
                auto_apply=False,  # 不自动应用，需要人工确认
                coordinator=self.component_coordinator  # 连接到ComponentCoordinator
            )
            print("   [System] 🔧 Bridge Auto Repair Online - Self-healing enabled (manual confirmation mode).")
        except Exception as e:
            print(f"   [System] ⚠️ Bridge Auto Repair initialization failed: {e}")

        # 🆕 [2026-01-11] Initialize M1-M4 Fractal AGI Components Adapter
        # 集成递归自指分形AGI的四个核心组件
        self.m1m4_adapter = None
        try:
            from core.m1m4_adapter import create_m1m4_adapter
            self.m1m4_adapter = create_m1m4_adapter(
                event_bus=self.event_bus,
                project_root=os.getcwd()
            )
            # 暴露自修改引擎到系统（用于集成级回归与能力激活）
            if hasattr(self.m1m4_adapter, 'self_modifier'):
                self.self_modifier = self.m1m4_adapter.self_modifier
                if self.self_modifier:
                    self.component_coordinator.register_component("self_modification", self.self_modifier)
                    print("   [System] 🧰 Self-Modification Engine registered in ComponentCoordinator")
                    # 🆕 [2026-01-17] 确保 tool_bridge 能访问 self_modifier
                    if self.tool_bridge and hasattr(self.tool_bridge, 'agi_system'):
                        # tool_bridge 已经持有 self 引用，现在 self.self_modifier 可用了
                        print("   [System] 🔗 Tool Bridge linked to Self-Modification Engine")
            print("   [System] 🧬 M1-M4 Fractal AGI Components Integrated:")
            print("      [M1] MetaLearner - 元参数优化器")
            print("      [M2] GoalQuestioner - 目标质疑模块")
            print("      [M3] SelfModifyingEngine - 架构自修改引擎")
            print("      [M4] RecursiveSelfMemory - 递归自引用记忆系统")
        except Exception as e:
            print(f"   [System] ⚠️ M1-M4 Adapter initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # [2026-01-11] Intelligence Upgrade: Short-term Working Memory
        # 短期工作记忆 - 打破思想循环，提升推理连贯性
        logging.info("   [DEBUG] About to initialize Working Memory...")
        print("   [DEBUG] About to initialize Working Memory...", flush=True)
        self.working_memory = None
        try:
            from core.working_memory import ShortTermWorkingMemory
            logging.info("   [DEBUG] Working Memory module imported, creating instance...")
            print("   [DEBUG] Working Memory module imported, creating instance...", flush=True)
            self.working_memory = ShortTermWorkingMemory(capacity=7, loop_threshold=3)
            self.intelligence_upgrade_enabled = True
            logging.info("   [System] [Intelligence Upgrade] Short-term Working Memory enabled")
            print("   [System] [Intelligence Upgrade] Short-term Working Memory enabled", flush=True)
        except Exception as e:
            logging.warning(f"   [System] [WARNING] Working memory initialization failed: {e}")
            print(f"   [System] [WARNING] Working memory initialization failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.intelligence_upgrade_enabled = False

        # 🆕 [2026-01-16] P0修复：熵值调节器 - 维持长期中熵状态
        logging.info("   [DEBUG] About to initialize Entropy Regulator...")
        print("   [DEBUG] About to initialize Entropy Regulator...", flush=True)
        self.entropy_regulator = None
        try:
            from core.entropy_regulator import EntropyRegulator
            logging.info("   [DEBUG] Entropy Regulator module imported, creating instance...")
            print("   [DEBUG] Entropy Regulator module imported, creating instance...", flush=True)
            # 🆕 [2026-01-17] P0修复：降低阈值以更早触发熵值调节
            self.entropy_regulator = EntropyRegulator(
                monitor_window=50,          # 缩短监控窗口以更快响应
                warning_threshold=0.6,       # 更早警告 (原0.7)
                critical_threshold=0.75,     # 更早触发临界调节 (原0.9)
                rising_threshold=5           # 更敏感的上升检测 (原10)
            )
            logging.info("   [System] [Entropy Regulation] Entropy Regulator enabled (enhanced)")
            print("   [System] [Entropy Regulation] Entropy Regulator enabled (enhanced thresholds)", flush=True)
        except Exception as e:
            logging.warning(f"   [System] [WARNING] Entropy Regulator initialization failed: {e}")
            print(f"   [System] [WARNING] Entropy Regulator initialization failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-17] 知识图谱实时导出器 - 支持可视化实时更新
        logging.info("   [DEBUG] About to initialize Knowledge Graph Exporter...")
        print("   [DEBUG] About to initialize Knowledge Graph Exporter...", flush=True)
        self.knowledge_graph_exporter = None
        try:
            from core.knowledge_graph_exporter import KnowledgeGraphExporter
            logging.info("   [DEBUG] Knowledge Graph Exporter module imported, creating instance...")
            print("   [DEBUG] Knowledge Graph Exporter module imported, creating instance...", flush=True)
            self.knowledge_graph_exporter = KnowledgeGraphExporter(
                output_dir="data/knowledge",
                export_interval=30,  # 每30秒导出一次
                max_history=100
            )
            # 启动自动导出线程
            self.knowledge_graph_exporter.start()
            logging.info("   [System] [Knowledge Graph Exporter] Knowledge Graph Exporter enabled")
            print("   [System] [Knowledge Graph Exporter] Knowledge Graph Exporter enabled", flush=True)
        except Exception as e:
            logging.warning(f"   [System] [WARNING] Knowledge Graph Exporter initialization failed: {e}")
            print(f"   [System] [WARNING] Knowledge Graph Exporter initialization failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # [2026-01-11] Intelligence Upgrade Phase 2: Reasoning Scheduler
        # 推理调度器 - 智能调度推理引擎，实现深度推理
        logging.info("   [DEBUG] About to initialize Reasoning Scheduler...")
        print("   [DEBUG] About to initialize Reasoning Scheduler...", flush=True)
        self.reasoning_scheduler = None
        
        # 🔧 RE-ENABLED [2026-01-12] Phase 2: Reasoning Scheduler
        try:
            logging.info("   [DEBUG] Attempting to import ReasoningScheduler...")
            from core.reasoning_scheduler import ReasoningScheduler
            logging.info("   [DEBUG] ReasoningScheduler module imported, importing CausalReasoningEngine...")
            from core.causal_reasoning import CausalReasoningEngine

            # 创建因果推理引擎
            logging.info("   [DEBUG] Creating CausalReasoningEngine instance...")
            print("   [DEBUG] Creating CausalReasoningEngine instance...", flush=True)
            causal_engine = CausalReasoningEngine()
            
            logging.info("   [DEBUG] CausalReasoningEngine created, creating ReasoningScheduler...")
            print("   [DEBUG] CausalReasoningEngine created, creating ReasoningScheduler...", flush=True)

            # 创建推理调度器
            self.reasoning_scheduler = ReasoningScheduler(
                causal_engine=causal_engine,
                llm_service=self.llm_service,
                confidence_threshold=0.6,
                max_depth=99999
            )
            logging.info("   [DEBUG] ReasoningScheduler created, starting session...")

            # 启动初始推理会话
            self.reasoning_scheduler.start_session()
            logging.info("   [DEBUG] Session started, Reasoning Scheduler initialization complete")

            print("   [System] [Intelligence Upgrade Phase 2] Reasoning Scheduler enabled (max_depth=99999)", flush=True)
        except Exception as e:
            print(f"   [System] [WARNING] Reasoning scheduler initialization failed: {e}", flush=True)
            logging.error(f"Reasoning scheduler initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # [2026-01-11] Intelligence Upgrade Phase 3: World Model, Goal Manager, Creative Exploration
        # 统一世界模型、层级目标系统、创造性探索引擎
        logging.info("   [DEBUG] About to initialize Phase 3 modules...")
        print("   [DEBUG] About to initialize Phase 3 modules...", flush=True)
        self.world_model = None
        self.creative_engine = None
        self.hierarchical_goal_manager = None

        try:
            from core.bayesian_world_model import BayesianWorldModel
            print("   [DEBUG] BayesianWorldModel imported, importing HierarchicalGoalManager...")
            from core.hierarchical_goal_manager import HierarchicalGoalManager, GoalLevel
            print("   [DEBUG] HierarchicalGoalManager imported, importing CreativeExplorationEngine...")
            from core.creative_exploration_engine import CreativeExplorationEngine

            # 创建世界模型
            self.world_model = BayesianWorldModel(learning_rate=0.1)
            print("   [System] [Intelligence Upgrade Phase 3] Bayesian World Model enabled")

            # 创建目标管理器
            self.hierarchical_goal_manager = HierarchicalGoalManager(max_active_goals=10)
            # 创建初始终身目标
            self.hierarchical_goal_manager.create_goal(
                name="achieve_agi",
                level=GoalLevel.LIFETIME,
                description="实现通用人工智能",
                priority=1.0
            )
            print("   [System] [Intelligence Upgrade Phase 3] Hierarchical Goal Manager enabled")

            # 创建创造性探索引擎
            self.creative_engine = CreativeExplorationEngine(temperature=0.7)
            print("   [System] [Intelligence Upgrade Phase 3] Creative Exploration Engine enabled")

        except Exception as e:
            print(f"   [System] [WARNING] Phase 3 modules initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # [2026-01-11] Intelligence Upgrade Phase 4: Meta-Learning, Self-Improvement, Recursive Self-Reference
        # 元学习、自我改进、递归自指优化
        print("   [DEBUG] About to initialize Phase 4 modules...")
        self.meta_learner = None
        self.self_improvement_engine = None
        self.recursive_self_reference = None

        try:
            from core.meta_learning import MetaLearner
            print("   [DEBUG] MetaLearner imported, importing SelfImprovementEngine...")
            from core.self_improvement import SelfImprovementEngine
            print("   [DEBUG] SelfImprovementEngine imported, importing RecursiveSelfReferenceEngine...")
            from core.recursive_self_reference import RecursiveSelfReferenceEngine

            # 创建元学习引擎
            self.meta_learner = MetaLearner(memory_size=100)
            print("   [System] [Intelligence Upgrade Phase 4] Meta-Learner enabled")

            # 创建自我改进引擎
            project_root = os.path.dirname(os.path.abspath(__file__))
            self.self_improvement_engine = SelfImprovementEngine(project_root)
            print("   [System] [Intelligence Upgrade Phase 4] Self-Improvement Engine enabled")

            # 创建递归自指引擎
            self.recursive_self_reference = RecursiveSelfReferenceEngine(max_recursion_depth=3)
            print("   [System] [Intelligence Upgrade Phase 4] Recursive Self-Reference enabled")

        except Exception as e:
            print(f"   [System] [WARNING] Phase 4 modules initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-16] P0修复: 元认知层 - 让系统"思考自己的思考"
        # 这是实现Level 4智能（元认知）的关键组件
        self.meta_cognitive_layer = None
        try:
            from core.meta_cognitive import MetaCognitiveLayer
            self.meta_cognitive_layer = MetaCognitiveLayer(
                knowledge_graph=self.memory,
                memory_system=self.semantic_memory
            )
            print("   [System] 🧠 Meta-Cognitive Layer Online - Self-Reflection Enabled")
            print("      [MetaCog] ✅ Task Understanding Depth Evaluator")
            print("      [MetaCog] ✅ Capability Matcher")
            print("      [MetaCog] ✅ Failure Attribution Engine")
        except Exception as e:
            print(f"   [System] ⚠️ Meta-Cognitive Layer initialization failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 🆕 [2026-01-30] P0修复: 元认知智能过滤器 - 解决空转和假阳性问题
        self.meta_filter = None
        try:
            from core.metacognitive_filter import get_meta_filter
            self.meta_filter = get_meta_filter()
            print("   [System] 🧠 Meta-Cognitive Filter Online - Smart Evaluation Enabled")
            print("      [MetaFilter] ✅ Complexity Threshold Filter")
            print("      [MetaFilter] ✅ Cooldown Mechanism")
            print("      [MetaFilter] ✅ Duplicate Detection")
            print("      [MetaFilter] ✅ Monitoring Task Whitelist")
        except Exception as e:
            print(f"   [System] ⚠️ Meta-Cognitive Filter initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-30] P1修复: 复杂任务生成器 - 解决推理深度停滞问题
        self.complex_task_generator = None
        try:
            from core.complex_task_generator import create_complex_task_generator
            self.complex_task_generator = create_complex_task_generator()
            print("   [System] 🎯 Complex Task Generator Online")
            print("      [TaskGen] ✅ Creative Tool Templates")
            print("      [TaskGen] ✅ Deep Analysis Templates")
            print("      [TaskGen] ✅ Cross-Domain Templates")
        except Exception as e:
            print(f"   [System] ⚠️ Complex Task Generator initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-30] P0修复: 创造性产出流水线 - 解决0产出问题
        self.creative_pipeline = None
        try:
            from core.creative_output_pipeline import create_creative_pipeline
            self.creative_pipeline = create_creative_pipeline()
            print("   [System] 🚀 Creative Output Pipeline Online")
            print("      [Pipeline] ✅ 5-Stage Process")
            print("      [Pipeline] ✅ Auto-Repair Mechanism")
            print("      [Pipeline] ✅ Quality Scoring")
        except Exception as e:
            print(f"   [System] ⚠️ Creative Pipeline initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-30] P2修复: 真进化机制引擎 - 架构自修改能力
        self.evolution_engine = None
        try:
            from core.true_evolution_engine import create_evolution_engine
            self.evolution_engine = create_evolution_engine(project_root)
            print("   [System] 🧬 True Evolution Engine Online")
            print("      [Evolution] ✅ Isolated Sandbox")
            print("      [Evolution] ✅ Automated Testing")
            print("      [Evolution] ✅ Version Control & Rollback")
        except Exception as e:
            print(f"   [System] ⚠️ True Evolution Engine initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-30] P2修复: 模块精简重构系统
        self.module_restructuring = None
        try:
            from core.module_restructuring import analyze_and_plan_restructuring
            self.module_restructuring = analyze_and_plan_restructuring(project_root)
            print("   [System] 🏗️  Module Restructuring System Online")
            print("      [Restructure] ✅ Module Analysis")
            print("      [Restructure] ✅ Legacy Detection")
            print("      [Restructure] ✅ Consolidation Planning")
            # 导出重构计划
            self.module_restructuring.export_plan("data/module_restructuring_plan.json")
        except Exception as e:
            print(f"   [System] ⚠️ Module Restructuring initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-16] P0修复: 架构感知层 - 让系统"理解自己的架构")
        # 这是实现Level 4智能（架构自我认知）的关键组件
        self.architecture_awareness_layer = None
        try:
            from core.architecture_awareness import ArchitectureAwarenessLayer
            self.architecture_awareness_layer = ArchitectureAwarenessLayer(
                project_root=os.getcwd()
            )
            print("   [System] 🏗️  Architecture Awareness Layer Online - Self-Understanding Enabled")
            print("      [ArchAware] ✅ Component Dependency Mapper")
            print("      [ArchAware] ✅ Performance Bottleneck Analyzer")
            print("      [ArchAware] ✅ Architecture Health Monitor")
        except Exception as e:
            print(f"   [System] ⚠️ Architecture Awareness Layer initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-17] 启动钩子 - 自动加载文档索引和执行启动任务
        self.startup_hooks = None
        try:
            from core.startup_hooks import StartupHooks
            self.startup_hooks = StartupHooks(
                knowledge_graph=self.memory,
                llm_service=self.llm_service
            )
            # 执行所有启动钩子
            startup_result = self.startup_hooks.execute_all()
            if startup_result.get("status") != "disabled":
                task_count = len(startup_result.get("tasks", []))
                print(f"   [System] 🚀 Startup Hooks Complete - {task_count} tasks executed")
        except Exception as e:
            print(f"   [System] ⚠️ Startup Hooks initialization failed: {e}")
            import traceback
            traceback.print_exc()

        self.error_diagnosis = None # State to hold error info for the next planning cycle
        self._meta_plugins = {}
        self._hot_swapper = None

        # 🆕 [2026-01-18] 自主性激活层 - 让现有组件"活"起来
        # 核心突破：将组件从"被动响应"模式转换为"主动驱动"模式
        self.autonomy_activator = None
        self.tool_factory = None
        try:
            # 初始化 ToolFactory (动态工具创建能力)
            from agi_tool_factory import ToolFactory, ToolRegistry
            from agi_dynamic_loader import DynamicModuleLoader
            
            tool_loader = DynamicModuleLoader(safe_mode=True)
            tool_registry = ToolRegistry()
            self.tool_factory = ToolFactory(tool_loader, tool_registry, coordinator=self.component_coordinator)
            print("   [System] 🔧 ToolFactory initialized - Dynamic tool creation enabled")
            
            # 初始化 AutonomyActivator (自主性激活层)
            from core.autonomy_activator import create_autonomy_activator
            
            self.autonomy_activator = create_autonomy_activator(
                goal_manager=self.goal_manager,
                m1m4_adapter=self.m1m4_adapter,
                tool_factory=self.tool_factory,
                event_bus=self.event_bus,
                biological_memory=self.biological_memory
            )
            
            print("   [System] 🔋 Autonomy Activator Online - Components ACTIVELY driven")
            print("      [Autonomy] ✅ GoalQuestioner - Will QUESTION goals every 50 ticks")
            print("      [Autonomy] ✅ IntrinsicMotivation - Will COMPUTE motivation every 10 ticks")
            print("      [Autonomy] ✅ ToolFactory - Will CREATE tools when capability gaps detected")
            
        except Exception as e:
            print(f"   [System] ⚠️ Autonomy Activator initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🆕 [2026-01-19] 系统优化器 - 零拓扑改动方案
        # 激活现有系统已实现但未充分利用的能力
        self.system_optimizer = None
        try:
            from core.system_optimizer import SystemOptimizer
            self.system_optimizer = SystemOptimizer(agi_engine=self)

            # 检查是否在启动时自动应用优化
            auto_optimize = '--optimize-on-startup' in sys.argv

            if auto_optimize:
                print("   [System] 🚀 SystemOptimizer Online - Applying optimizations on startup...")
                results = self.system_optimizer.apply_all_optimizations()
                print(f"   [System] ✅ {len(results)} optimizations applied")
            else:
                print("   [System] 🔧 SystemOptimizer Online - Ready (use --optimize-on-startup to activate)")
                print("      [Optimizer] 💡 Available optimizations:")
                print("         - Creativity Emergence: 0.04 → 0.15 (+275%)")
                print("         - Deep Reasoning: 100 steps → 99,999 steps (+999x)")
                print("         - Autonomous Goals: Generation rate × 2")
                print("         - Cross-Domain Transfer: Auto-activate (+18.3%)")

        except Exception as e:
            print(f"   [System] ⚠️ SystemOptimizer initialization failed: {e}")
            import traceback
            traceback.print_exc()

        # 🔧 [2026-01-23] P0修复: 集成断裂的拓扑连接组件
        # 修复3D拓扑图中显示但未在引擎中初始化的关键组件

        # 1. 自主目标系统 (AutonomousGoalSystem)
        self.autonomous_goal_system = None
        try:
            from core.autonomous_goal_system import AutonomousGoalGenerator
            self.autonomous_goal_system = AutonomousGoalGenerator()
            print("   [System] 🎯 AutonomousGoalSystem Online - 自主目标生成已启用")
            print("      [AutoGoal] ✅ IntrinsicValueFunction - 内在价值计算")
            print("      [AutoGoal] ✅ OpportunityRecognitionEngine - 机会识别")
            print("      [AutoGoal] ✅ GoalHierarchyBuilder - 目标层级构建")
        except Exception as e:
            print(f"   [System] ⚠️ AutonomousGoalSystem initialization failed: {e}")

        # 2. 认知桥接器 (CognitiveBridge)
        self.cognitive_bridge = None
        try:
            from core.cognitive_bridge import CognitiveBridge
            self.cognitive_bridge = CognitiveBridge(agi_engine=self)
            print("   [System] 🧠 CognitiveBridge Online - 认知能力桥接已启用")
            print("      [CogBridge] ✅ TopologyMemory Query - 拓扑记忆查询")
            print("      [CogBridge] ✅ CausalReasoning Query - 因果推理查询")
            print("      [CogBridge] ✅ DeepReasoning Integration - 深度推理集成")

            # 🆕 [ENHANCE 2026-01-27] 更新intent_bridge的cognitive_bridge引用
            if self.intent_bridge is not None:
                # 🆕 [ENHANCE 2026-01-27] 同时保持Redis IPC启用
                self.intent_bridge = get_intent_bridge(
                    cognitive_bridge=self.cognitive_bridge,
                    use_redis=True,  # 🆕 保持Redis IPC启用
                    redis_host='localhost',
                    redis_port=6379
                )
                print("   [System] 🔗 IntentBridge updated with CognitiveBridge - 深度分析已启用")

                # 🆕 [EVENT FLOW 2026-01-27] 订阅Redis Pub/Sub事件流
                if hasattr(self.intent_bridge, 'subscribe_events'):
                    success = self.intent_bridge.subscribe_events(self._handle_agi_event)
                    if success:
                        print("   [System] 🎧 Event Subscription Online - 事件驱动已启用")

                # 🆕 [ADAPTIVE POLLING 2026-01-27] 初始化自适应轮询器
                if self._adaptive_poller is None:
                    from intent_dialogue_bridge import AdaptivePoller
                    self._adaptive_poller = AdaptivePoller()
                    print("   [System] 🔄 Adaptive Poller Online - 自适应轮询已启用")
        except Exception as e:
            print(f"   [System] ⚠️ CognitiveBridge initialization failed: {e}")

        # 3. 跨域迁移系统 (CrossDomainTransfer)
        self.cross_domain_transfer = None
        try:
            from core.cross_domain_transfer import CrossDomainTransferSystem
            self.cross_domain_transfer = CrossDomainTransferSystem()
            print("   [System] 🔄 CrossDomainTransfer Online - 跨域知识迁移已启用")
            print("      [Xfer] ✅ CrossDomainMapper - 跨域映射")
            print("      [Xfer] ✅ MetaLearningTransfer - 元学习迁移")
            print("      [Xfer] ✅ FewShotLearner - 少样本学习")
            print("      [Xfer] ✅ SkillExtractor - 技能提取")
        except Exception as e:
            print(f"   [System] ⚠️ CrossDomainTransfer initialization failed: {e}")

        # 🔧 [2026-01-23] 建立组件间的拓扑连接
        # 修复信息流和事件流断裂
        self._establish_component_connections()

        # 🆕 [2026-01-24] 会话上下文恢复器 - 修复会话连续性问题
        # 自动恢复历史对话上下文和未完成任务，减少重复解释
        self.session_restorer = None
        try:
            from core.session_context_restorer import SessionContextRestorer
            self.session_restorer = SessionContextRestorer(project_root=os.getcwd())

            # 尝试恢复上一次会话的上下文
            restored_context = self.session_restorer.restore_context()
            if restored_context.get("restoration_success"):
                active_goals_count = len(restored_context.get("active_goals", []))
                recent_insights_count = len(restored_context.get("recent_insights", []))
                print(f"   [System] 🔄 Session Context Restorer Online - 跨会话连续性已启用")
                print(f"      [ContextRestore] ✅ 恢复了 {active_goals_count} 个活跃目标")
                print(f"      [ContextRestore] ✅ 恢复了 {recent_insights_count} 条最近洞察")
                if restored_context.get("last_session_tasks"):
                    pending_count = len(restored_context.get("last_session_tasks", []))
                    print(f"      [ContextRestore] ⏳ {pending_count} 个待处理任务已识别")
            else:
                print(f"   [System] 🔄 Session Context Restorer Online - 首次启动或无历史上下文")
        except Exception as e:
            print(f"   [System] ⚠️ Session Context Restorer initialization failed: {e}")
            import traceback
            traceback.print_exc()

    def _establish_component_connections(self):
        """
        建立组件间的拓扑连接，修复信息流和事件流断裂

        连接拓扑:
        AutonomousGoalSystem → DoubleHelixEngineV2
        CognitiveBridge → FractalIntelligence
        CrossDomainTransfer → KnowledgeGraph
        MetaCognitiveLayer → DoubleHelixEngineV2
        """
        try:
            connections_established = 0

            # 连接1: AutonomousGoalSystem → DoubleHelixEngineV2
            # 目标生成流：自主目标 → 双螺旋决策引擎
            if self.autonomous_goal_system and self.helix_engine:
                # 注册事件处理器：目标生成完成时通知双螺旋引擎
                self.event_bus.subscribe("autonomous_goal_generated", self._on_autonomous_goal_generated)
                connections_established += 1
                print("   [Topology] ✅ Connected: AutonomousGoalSystem → DoubleHelixEngineV2")

            # 连接2: CognitiveBridge → FractalIntelligence (through helix_engine)
            # 认知验证流：认知桥接 → 分形智能
            if self.cognitive_bridge and self.helix_engine:
                # 认知桥接器可以为分形智能提供拓扑和因果分析
                if hasattr(self.helix_engine, 'fractal') and self.helix_engine.fractal:
                    # 将认知桥接注入到FractalIntelligence
                    if hasattr(self.helix_engine.fractal, 'set_cognitive_bridge'):
                        self.helix_engine.fractal.set_cognitive_bridge(self.cognitive_bridge)
                    connections_established += 1
                    print("   [Topology] ✅ Connected: CognitiveBridge → FractalIntelligence")

            # 连接3: CrossDomainTransfer → KnowledgeGraph
            # 跨域迁移流：知识图谱 → 跨域迁移 → 知识图谱
            if self.cross_domain_transfer and self.memory:
                # 将知识图谱注入到跨域迁移系统
                # 这里我们保存引用，在运行时动态使用
                if not hasattr(self.cross_domain_transfer, 'knowledge_graph'):
                    self.cross_domain_transfer.knowledge_graph = self.memory
                connections_established += 1
                print("   [Topology] ✅ Connected: CrossDomainTransfer ↔ KnowledgeGraph")

            # 连接4: MetaCognitiveLayer → DoubleHelixEngineV2
            # 自我评估流：元认知层 → 双螺旋引擎
            if self.meta_cognitive_layer and self.helix_engine:
                # 注册事件处理器：元认知评估完成时通知双螺旋引擎
                self.event_bus.subscribe("meta_cognitive_assessment", self._on_meta_cognitive_assessment)
                connections_established += 1
                print("   [Topology] ✅ Connected: MetaCognitiveLayer → DoubleHelixEngineV2")

            print(f"   [Topology] 🧬 拓扑连接建立完成: {connections_established} 条连接")

        except Exception as e:
            print(f"   [Topology] ⚠️ 建立拓扑连接时出错: {e}")
            import traceback
            traceback.print_exc()

    async def _on_autonomous_goal_generated(self, event_data: dict):
        """
        处理自主目标生成事件
        将自主生成的目标注入到双螺旋引擎
        """
        try:
            goal = event_data.get('goal')
            if goal and self.helix_engine:
                # 将目标传递给双螺旋引擎
                print(f"   [EventFlow] 🎯 自主目标注入: {goal.get('description', 'Unknown')}")
                # 这里可以进一步处理，比如调整螺旋参数等
        except Exception as e:
            print(f"   [EventFlow] ⚠️ 处理自主目标事件失败: {e}")

    async def _on_meta_cognitive_assessment(self, event_data: dict):
        """
        处理元认知评估事件
        将元认知评估结果反馈给双螺旋引擎
        """
        try:
            assessment = event_data.get('assessment')
            if assessment and self.helix_engine:
                # 将评估结果反馈给双螺旋引擎
                print(f"   [EventFlow] 🧠 元认知反馈: 置信度={assessment.get('confidence', 0):.2f}")
                # 这里可以进一步处理，比如调整决策权重等
        except Exception as e:
            print(f"   [EventFlow] ⚠️ 处理元认知评估事件失败: {e}")

    async def _process_llm_response_with_bridge(self, llm_response: str) -> str:
        """
        通过桥接层处理 LLM 响应，执行其中的工具调用。
        
        Args:
            llm_response: LLM 的原始响应文本
            
        Returns:
            处理后的响应文本（包含工具执行结果）
        """
        if not self.tool_bridge or not llm_response:
            return llm_response
        
        # 检测是否包含工具调用
        if "TOOL_CALL:" in llm_response or self._contains_tool_pattern(llm_response):
            try:
                print("   [Bridge] 🔧 检测到工具调用，执行中...")
                result = await self.tool_bridge.process_response(llm_response)
                if result.get('has_tool_calls'):
                    tool_count = len(result.get('tool_results', []))
                    success_count = sum(1 for r in result.get('tool_results', []) if r.get('result', {}).get('success'))
                    print(f"   [Bridge] ✅ 工具执行完成: {success_count}/{tool_count} 成功")
                    return result['final_response']
            except Exception as e:
                print(f"   [Bridge] ⚠️ 工具执行出错: {e}")
        
        return llm_response
    
    def _contains_tool_pattern(self, response: str) -> bool:
        """检测响应是否包含工具调用模式"""
        patterns = [
            r'file_operation[s]?\.\w+\(',
            r'world_model\.\w+\(',
            r'knowledge_graph\.\w+\(',
            r'autonomous_document_create\.\w+\(',
            r'metacognition\.\w+\(',
            r'system_tools\.\w+\(',
            r'self_modif\w*\.\w+\(',  # 🆕 自修改工具模式
            r'code_patch\.\w+\(',     # 🆕 代码补丁模式
        ]
        for pattern in patterns:
            if re.search(pattern, response):
                return True
        return False
    
    # ========================================================================
    # 🆕 [2026-01-10] 系统依赖图构建（供增强验证器使用）
    # ========================================================================
    
    def _build_system_dependency_graph(self) -> dict:
        """
        构建系统函数依赖图
        
        扫描AGI系统中已存在的公开函数，供验证器判断洞察代码是否调用了
        不存在的函数（伪代码检测）。
        
        返回:
            {函数名: True} 的字典
        """
        graph = {}
        
        # 1. 注册核心模块的公开函数
        # 注意：该方法在 __init__ 早期被调用，部分组件可能尚未初始化
        core_objects = [
            ('memory', getattr(self, 'memory', None)),
            ('semantic_memory', getattr(self, 'semantic_memory', None)),
            ('biological_memory', getattr(self, 'biological_memory', None)),
            ('goal_manager', getattr(self, 'goal_manager', None)),
            ('motivation', getattr(self, 'motivation', None)),
            ('skill_manager', getattr(self, 'skill_manager', None)),
        ]
        
        for name, obj in core_objects:
            if obj:
                for attr_name in dir(obj):
                    if not attr_name.startswith('_'):
                        attr = getattr(obj, attr_name, None)
                        if callable(attr):
                            graph[attr_name] = True
        
        # 2. 注册全局工具函数
        global_funcs = [
            'save_insight_markdown',
            'log_cycle_flow',
            'parse_insight_file',
        ]
        for func_name in global_funcs:
            graph[func_name] = True
        
        # 3. 注册常用的系统辅助函数
        utility_funcs = [
            'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
            'min', 'max', 'sum', 'sorted', 'reversed', 'abs', 'round',
            'isinstance', 'hasattr', 'getattr', 'setattr',
            'dict', 'list', 'set', 'tuple', 'str', 'int', 'float', 'bool',
        ]
        for func_name in utility_funcs:
            graph[func_name] = True

        # 🆕 [2026-01-15] 4. 注册数学函数（修复Insight依赖缺失问题）
        math_funcs = [
            'exponential', 'exp',  # 指数函数（来自numpy/math）
            'sqrt', 'square',     # 平方根/平方
            'log', 'log10', 'log2',  # 对数函数
            'sin', 'cos', 'tan',   # 三角函数
            'pow', 'power',        # 幂函数
            'mean', 'median', 'std',  # 统计函数（numpy）
        ]
        for func_name in math_funcs:
            graph[func_name] = True

        return graph

    async def _on_insight_generated(self, event: Event):
        try:
            data = event.data
            ts_float = data.get("timestamp", time.time())
            now_ts = time.time()
            node_id = data.get("node_id") or f"Insight_{ts_float}"
            if now_ts < getattr(self, "_insight_persist_backoff_until", 0.0):
                remaining = int(max(0.0, getattr(self, "_insight_persist_backoff_until", 0.0) - now_ts))
                print(f"   [System] ⚠️ Insight persistence backoff active ({remaining}s remaining).")
                persist_result = {"success": False, "error": "backoff_active"}
                file_path = None
                abs_path = None
                timestamp = int(ts_float)
            else:
                persist_result = save_insight_markdown(data)
                if not persist_result.get("success"):
                    failure_ts = getattr(self, "_insight_persist_failure_ts", 0.0)
                    if now_ts - failure_ts > 60.0:
                        self._insight_persist_failure_count = 0
                    self._insight_persist_failure_count = getattr(self, "_insight_persist_failure_count", 0) + 1
                    self._insight_persist_failure_ts = now_ts
                    if self._insight_persist_failure_count >= 3:
                        self._insight_persist_backoff_until = now_ts + 60.0
                    print(f"   [System] ⚠️ Failed to save insight: {persist_result.get('error', 'insight persist failed')}")
                    file_path = None
                    abs_path = None
                    timestamp = int(ts_float)
                else:
                    file_path = persist_result["file_path"]
                    abs_path = persist_result["abs_path"]
                    timestamp = persist_result["timestamp"]
                    self._insight_persist_failure_count = 0
                    self._insight_persist_failure_ts = 0.0
                    self._insight_persist_backoff_until = 0.0

                if file_path:
                    print(f"   [System] 📝 Insight saved to {file_path}")
            
            # 2. Update Knowledge Graph Node with File Path (Topology Consistency)
            # Ensure the node in memory points to the physical file
            try:
                attrs = {
                    "persist_status": "ok" if abs_path else "failed",
                    "persist_error": None if abs_path else persist_result.get("error")
                }
                if abs_path:
                    attrs["file_path"] = abs_path
                # 🆕 [2026-01-30] P1修复: 使用孤立节点预防
                if self.isolation_prevention:
                    self.isolation_prevention.add_node_with_prevention(node_id, **attrs)
                else:
                    self.memory.graph.add_node(node_id, **attrs)
            except Exception as e:
                print(f"   [System] ⚠️ Failed to update insight node topology: {e}")

            # 3. Internalize into Biological Memory (Fluid Topology)
            try:
                if persist_result.get("error") == "backoff_active":
                    return
                epochs = 50 if abs_path else 10
                print(f"   [BioMemory] 🧠 Internalizing insight into Fluid Topology...")
                insight_item = {
                    "id": f"insight_{timestamp}",
                    "content": data.get("content", "") or "",
                    "source": "event_bus",
                    "persisted": bool(abs_path)
                }
                # Internalize (consolidate) this single insight immediately
                # This triggers topological growth and connection
                stats = self.biological_memory.internalize([insight_item], epochs=epochs)
                print(f"   [BioMemory] ✅ Insight internalized. Loss: {stats.get('final_loss', 0.0):.4f}")
                print(f"   [BioMemory]    Topology Size: {self.biological_memory.topology.size()} nodes")
            except Exception as e:
                print(f"   [BioMemory] ⚠️ Internalization failed: {e}")
            
        except Exception as e:
            print(f"   [System] ⚠️ Failed to save insight: {e}")

    # ========================================================================
    # 馃啎 [2026-01-15] 鍙岃灪鏃嬪喅绛栧紩鎿庨泦鎴愭柟娉?
    # ========================================================================

    def _encode_helix_state(self, context: Dict[str, Any]) -> 'np.ndarray':
        """
        [REFACTORED 2026-01-29] Real Semantic Encoding
        Uses PerceptionSystem (SentenceTransformers) instead of MD5.
        Generates a 64-dim state vector that captures TRUE meaning.
        """
        if hasattr(self, 'perception_system') and self.perception_system:
            # Inject scalar metrics for the encoder to append
            context['priority_score'] = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}.get(context.get('priority', 'medium'), 0.5)
            context['urgency'] = min(1.0, self.step_count / 1000.0)
            context['success_probability'] = context.get('success_probability', 0.5)
            
            # Use the real embedder
            return self.perception_system.encode_helix_state(context, target_dim=64)
        else:
            # Fallback (Should rarely happen if init succeeded)
            import numpy as np
            return np.zeros(64, dtype=np.float32)
    
    async def _helix_enhanced_decision(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用双螺旋引擎增强决策。
        
        返回：
        - enhanced_action: 增强后的动作建议
        - helix_confidence: 双螺旋置信度
        - fusion_method: 使用的融合方法
        - emergence_score: 涌现评分
        """
        if not self.helix_decision_enabled or self.helix_engine is None:
            return {
                'enhanced': False,
                'reason': 'Helix engine not available'
            }
        
        try:
            # 编码当前状态
            state_vector = self._encode_helix_state(action_context)
            
            # 调用双螺旋引擎
            helix_result = self.helix_engine.decide(state_vector)
            
            # 提取决策信息
            enhanced_info = {
                'enhanced': True,
                'helix_action': helix_result.action,
                'helix_confidence': helix_result.confidence,
                'fusion_method': helix_result.fusion_method,
                'emergence_score': helix_result.emergence_score,
                'system_a_conf': helix_result.system_a_confidence,
                'system_b_conf': helix_result.system_b_confidence,
                'complementary_preference': helix_result.complementary_preference,
                'reasoning': helix_result.reasoning
            }
            
            # 如果是创造性融合动作（4-7），标记为创造性决策
            if helix_result.action >= 4:
                enhanced_info['is_creative'] = True
                enhanced_info['creative_action_name'] = self._get_creative_action_name(helix_result.action)
            else:
                enhanced_info['is_creative'] = False
            
            return enhanced_info
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'enhanced': False,
                'reason': f'Helix decision failed: {e}'
            }
    
    def _get_creative_action_name(self, action_id: int) -> str:
        """获取创造性动作名称"""
        creative_actions = {
            4: 'stop_and_observe',
            5: 'explore_alternative',
            6: 'synthesize_novel',
            7: 'meta_reflect'
        }
        return creative_actions.get(action_id, f'creative_{action_id}')

    async def _process_intent_bridge(self):
        """
        处理来自意图桥接的用户意图。
        实现深度意图理解和确认流程。
        
        IntentDialogueBridge API：
        - poll_new_intent() -> Optional[Intent]
        - analyze_intent(intent: Intent) -> Intent  # 修改intent的深度属性
        - generate_confirmation(intent: Intent) -> Intent  # 修改intent状态
        - send_confirmation_request(intent: Intent)  # 发送确认请求
        - lock_attention(intent: Intent)  # 锁定注意力
        - send_execution_result(intent: Intent, result: str, success: bool)
        """
        if not self.intent_bridge:
            return

        # 🆕 [ADAPTIVE POLLING 2026-01-27] 使用自适应轮询策略
        should_poll = True  # 是否应该轮询
        poll_timeout = 1.0  # 默认超时

        if self._adaptive_poller and hasattr(self.intent_bridge, 'check_cli_status'):
            try:
                # 收集状态信息
                cli_status = self.intent_bridge.check_cli_status()
                cli_online = cli_status.get('online', True)
                queue_length = self.intent_bridge.get_queue_length()

                # 计算轮询策略
                strategy = self._adaptive_poller.calculate_strategy(
                    cli_online=cli_online,
                    queue_length=queue_length
                )

                # 判断是否应该在本TICK轮询
                should_poll = self._adaptive_poller.should_poll_this_tick(
                    current_tick=self.step_count,
                    last_poll_tick=self._last_poll_tick,
                    strategy=strategy
                )

                # 应用策略
                if should_poll:
                    self._last_poll_tick = self.step_count
                    poll_timeout = strategy.timeout

                    # 记录策略变更（仅日志级别）
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"[AdaptivePoller] 🎯 策略: {strategy.mode} | "
                            f"间隔: {strategy.interval_ticks} TICK | "
                            f"超时: {strategy.timeout}s | "
                            f"原因: {strategy.reason}"
                        )
                else:
                    # 跳过本轮
                    if strategy.mode == "idle":
                        logger.debug(f"[AdaptivePoller] 💤 CLI离线，跳过轮询 (TICK {self.step_count})")
                    elif strategy.mode == "empty":
                        logger.debug(f"[AdaptivePoller] 📉 空轮询率过高，跳过本轮")

            except Exception as e:
                logger.warning(f"⚠️ 自适应轮询失败: {e}，使用默认策略")
                should_poll = True

        # 如果不应该轮询，直接返回
        if not should_poll:
            return

        try:
            # 检查是否有待执行的已确认意图
            current = self.intent_bridge.get_current_intent()
            if current:
                # 🔧 [FIX 2026-01-18] 先检查确认超时，避免永久阻塞
                self.intent_bridge._check_confirmation_timeout()

                # 重新获取当前意图（可能已因超时而改变）
                current = self.intent_bridge.get_current_intent()
                if not current:
                    # 意图已超时并自动确认，继续处理
                    pass
                elif current.state == IntentState.CONFIRMED:
                    print(f"   [IntentBridge] ▶️ 检测到已确认意图，开始执行: {current.id[:8]}...")
                    await self._execute_confirmed_intent(current)
                    return
                elif current.state == IntentState.CONFIRMING:
                    # 🆕 [FIX 2026-01-27] 检查是否超时
                    elapsed = time.time() - current.timestamp
                    timeout_seconds = 300  # 5分钟超时

                    if elapsed > timeout_seconds:
                        # 超时：自动标记为失败并继续处理
                        print(f"   [IntentBridge] ⏰ 意图确认超时 ({elapsed:.0f}s > {timeout_seconds}s): {current.id[:8]}...")
                        current.state = IntentState.FAILED
                        current.error_message = f"确认超时 ({elapsed:.0f}秒)"
                        self.intent_bridge.unlock_attention()

                        # 清除当前意图，继续处理pending队列
                        print(f"   [IntentBridge] 🔄 继续处理pending队列...")
                    else:
                        # 还在等待确认，但不阻塞pending队列
                        # 尝试处理pending队列中的非CONFIRMING意图
                        print(f"   [IntentBridge] ⏳ 意图等待确认中 ({elapsed:.0f}s)，尝试处理pending队列...")
                        # 继续执行，不return，允许处理pending队列
                elif current.state == IntentState.REJECTED:
                    # 意图被拒绝，清理并继续
                    print(f"   [IntentBridge] 🚫 意图已被用户拒绝: {current.id[:8]}...")
                    self.intent_bridge.unlock_attention()
                    return

            # 轮询新意图（使用自适应超时）
            intent = self.intent_bridge.poll_new_intent(timeout=poll_timeout)

            # 🆕 [ADAPTIVE POLLING 2026-01-27] 记录轮询结果
            if self._adaptive_poller:
                is_empty = (intent is None)
                self._adaptive_poller.record_poll_result(is_empty)

            if not intent:
                return
            
            print(f"   [IntentBridge] 📥 收到新意图: {intent.id[:8]}...")
            print(f"   [IntentBridge]    原文: {intent.raw_input[:100]}...")
            
            # 分析意图深度 - analyze_intent 接受 Intent 对象并返回修改后的 Intent
            intent = self.intent_bridge.analyze_intent(intent)
            
            print(f"   [IntentBridge] 🔍 意图分析完成:")
            print(f"   [IntentBridge]    深度: {intent.depth.value if intent.depth else 'UNKNOWN'}")
            print(f"   [IntentBridge]    表面请求: {intent.surface_request[:50]}...")
            print(f"   [IntentBridge]    深层目标: {intent.deep_goal[:50] if intent.deep_goal else 'None'}...")
            
            # 根据深度决定是否需要确认
            needs_confirmation = intent.depth in [IntentDepth.DEEP, IntentDepth.PHILOSOPHICAL]
            
            if needs_confirmation:
                # 生成确认请求 - generate_confirmation 接受 Intent 对象
                intent = self.intent_bridge.generate_confirmation(intent)
                
                # 发送确认请求给用户
                self.intent_bridge.send_confirmation_request(intent)
                
                print(f"   [IntentBridge] 🔄 等待用户确认...")
                
                # 等待用户确认（非阻塞，下一轮循环继续检查）
                return
            
            # 表面/中等深度意图，自动确认并执行
            intent.state = IntentState.CONFIRMED
            self.intent_bridge.lock_attention(intent)
            await self._execute_confirmed_intent(intent)
            
        except Exception as e:
            print(f"   [IntentBridge] ❌ 意图处理错误: {e}")
            import traceback
            traceback.print_exc()
            
            # ✅ [FIX 2026-01-09] 即使出错也要发送响应，避免CLI永久等待
            if hasattr(self, 'intent_bridge') and self.intent_bridge:
                try:
                    # 尝试获取当前意图
                    current_intent = self.intent_bridge.get_active_intent()
                    if current_intent:
                        self.intent_bridge.send_execution_result(
                            current_intent,
                            f"意图处理失败: {str(e)}",
                            success=False
                        )
                except Exception as send_err:
                    print(f"   [IntentBridge] ⚠️ 无法发送错误响应: {send_err}")

    # ==================== 🆕 事件处理机制 [EVENT FLOW 2026-01-27] ====================

    def _handle_agi_event(self, event_data: Dict[str, Any]) -> None:
        """
        处理来自Redis Pub/Sub的事件流

        注意：此方法在后台线程中执行，不应执行阻塞操作
             只记录事件日志，不破坏TICK驱动的自主节律

        Args:
            event_data: 事件数据字典
                - event_type: 事件类型
                - source: 事件来源
                - timestamp: 时间戳
                - client_id: 客户端ID
                - data: 事件数据
        """
        event_type = event_data.get('event_type')
        source = event_data.get('source', 'unknown')
        client_id = event_data.get('client_id', 'unknown')
        data = event_data.get('data', {})

        try:
            if event_type == 'intent_submitted':
                # 意图已提交事件
                intent_id = data.get('intent_id', 'unknown')[:8]
                queue_depth = data.get('queue_depth', 0)
                user_input = data.get('user_input', '')[:50]

                logger.info(f"[Event] 📨 意图已提交: {intent_id}... | 队列深度: {queue_depth}")
                logger.debug(f"[Event]    来源: {client_id} | 内容: {user_input}...")

                # 可选：如果队列深度较大，可以临时提升轮询优先级
                if queue_depth > 5:
                    logger.warning(f"[Event] ⚠️ 意图队列积压: {queue_depth}个意图待处理")

            elif event_type == 'intent_completed':
                # 意图完成事件（可选，未来扩展）
                intent_id = data.get('intent_id', 'unknown')[:8]
                success = data.get('success', False)
                logger.info(f"[Event] ✅ 意图已完成: {intent_id}... | 成功: {success}")

            elif event_type == 'cli_connected':
                # CLI连接事件
                logger.info(f"[Event] 💚 Chat CLI已上线: {client_id}")

            else:
                # 未知事件类型
                logger.debug(f"[Event] ❓ 未知事件类型: {event_type} | 来源: {source}")

        except Exception as e:
            logger.warning(f"⚠️ 事件处理失败: {e} | 事件: {event_type}")

    # ==================== 原有方法 ====================

    def _validate_plan_feasibility(self, plan: list) -> tuple:
        """
        验证计划的可行性 (Pre-Execution Validation)
        
        检查计划步骤是否包含未知的工具引用，防止幻觉调用。
        
        Returns:
            (is_feasible: bool, warning: str)
        """
        if not self.tool_bridge:
            return True, ""
            
        available_tools = self.tool_bridge.get_available_tools()
        # 转换为小写以便匹配
        tools_lower = {t.lower() for t in available_tools}
        
        warnings = []
        for i, step in enumerate(plan):
            step_lower = step.lower() if isinstance(step, str) else str(step).lower()
            
            # 检测工具调用模式: tool_name.method() 或 tool_name(
            tool_patterns = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*\w+\s*\(', step_lower)
            tool_patterns += re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', step_lower)
            
            for tool_name in tool_patterns:
                # 排除常见的非工具调用
                common_non_tools = {'print', 'str', 'int', 'float', 'list', 'dict', 'set', 
                                   'len', 'range', 'open', 'close', 'read', 'write',
                                   'async', 'await', 'self', 'return', 'if', 'for', 'while'}
                if tool_name in common_non_tools:
                    continue
                    
                # 检查是否是已注册工具
                if tool_name not in tools_lower:
                    # 检查是否是已注册工具的子串（如 file 对应 file_operation）
                    is_partial_match = any(tool_name in t or t.startswith(tool_name) for t in tools_lower)
                    if not is_partial_match and len(tool_name) > 3:
                        warnings.append(f"Step {i+1}: 未知工具 '{tool_name}'")
        
        if warnings:
            return False, "; ".join(warnings[:3])
        return True, ""

    def _show_intelligence_upgrade_status(self):
        """
        Display intelligence upgrade status (Phase 1: Working Memory)
        """
        if not hasattr(self, 'working_memory') or not self.working_memory:
            print("   [Intelligence Upgrade] Not enabled")
            return

        summary = self.working_memory.get_context_summary()
        stats = summary['stats']

        print("\n   [Intelligence Upgrade Status]")
        print(f"   Active thoughts: {summary['active_thoughts_count']}/{self.working_memory.capacity}")
        print(f"   Current action: {summary['current_action']}")
        print(f"   Thought diversity: {summary['diversity']:.2f}")
        print(f"   Total thoughts: {stats['total_thoughts']}")
        print(f"   Loops detected: {stats['loops_detected']}")
        print(f"   Loops broken: {stats['loops_broken']}")
        print(f"   Divergent thoughts: {stats['divergent_thoughts']}")
        
        # 🆕 [2026-01-30] P0修复: 工作记忆优化器统计
        if hasattr(self, '_working_memory_optimizer') and self._working_memory_optimizer:
            opt_stats = self._working_memory_optimizer.get_stats()
            print("\n   [Working Memory Optimizer Status]")
            print(f"   Cache hit rate: {opt_stats['hit_rate']:.2%}")
            print(f"   Cache size: {opt_stats['cache_size']}/{opt_stats.get('max_size', 1000)}")
            print(f"   Total requests: {opt_stats['total_requests']}")
            print(f"   Cache hits: {opt_stats['cache_hits']}")
            print(f"   LRU evictions: {opt_stats['evictions']}")

    def _show_reasoning_status(self):
        """
        Display reasoning scheduler status (Phase 2: Deep Reasoning)
        """
        if not hasattr(self, 'reasoning_scheduler') or not self.reasoning_scheduler:
            print("   [Reasoning Scheduler] Not enabled")
            return

        print("\n   [Reasoning Scheduler Status]")

        # Get current session summary
        session_summary = self.reasoning_scheduler.get_current_session_summary()
        stats = self.reasoning_scheduler.get_statistics()

        if session_summary:
            print(f"   Session ID: {session_summary.get('session_id', 'N/A')}")
            print(f"   Total steps: {session_summary.get('total_steps', 0)}")
            print(f"   Current depth: {session_summary.get('max_depth', 0)}/{self.reasoning_scheduler.max_depth}")
            print(f"   Avg confidence: {session_summary.get('avg_confidence', 0):.2f}")
            print(f"   Avg step time: {session_summary.get('avg_step_time', 0):.3f}s")

            # Mode distribution
            mode_dist = session_summary.get('mode_distribution', {})
            if mode_dist:
                print(f"   Reasoning modes:")
                for mode, count in mode_dist.items():
                    print(f"     - {mode}: {count}")

        # Overall statistics
        print(f"\n   Overall Statistics:")
        print(f"   Total reasoning calls: {stats.get('total_reasoning_calls', 0)}")
        print(f"   Causal reasoning used: {stats.get('causal_reasoning_used', 0)}")
        print(f"   LLM fallback used: {stats.get('llm_fallback_used', 0)}")
        print(f"   Hybrid reasoning used: {stats.get('hybrid_reasoning_used', 0)}")
        print(f"   Max depth achieved: {stats.get('max_depth_achieved', 0)}")
        print(f"   Causal ratio: {stats.get('causal_ratio', 0):.2%}")

        # Recent reasoning chain
        chain = self.reasoning_scheduler.get_reasoning_chain(n=5)
        if chain:
            print(f"\n   Recent reasoning chain (last {len(chain)} steps):")
            for step in chain:
                print(f"     [{step['step']}] {step['mode']} - depth={step['depth']}, conf={step['confidence']:.2f}")

    def _show_world_model_status(self):
        """Display world model status (Phase 3)"""
        if not hasattr(self, 'world_model') or not self.world_model:
            print("   [World Model] Not enabled")
            return

        print("\n   [Bayesian World Model Status]")

        summary = self.world_model.get_state_summary()

        print(f"   Total beliefs: {summary['total_beliefs']}")
        print(f"   Causal links: {summary['total_causal_links']}")
        print(f"   Interventions: {summary['total_interventions']}")
        print(f"   Avg confidence: {summary['avg_belief_confidence']:.3f}")
        print(f"   High confidence beliefs: {summary['high_confidence_beliefs']}")

        # Show sample beliefs
        beliefs = self.world_model.get_all_beliefs()
        if beliefs:
            print(f"\n   Sample beliefs (first 5):")
            for i, belief in enumerate(list(beliefs.values())[:5], 1):
                print(f"     {i}. {belief}")

    def _show_goal_manager_status(self):
        """Display goal manager status (Phase 3)"""
        if not hasattr(self, 'goal_manager') or not self.goal_manager:
            print("   [Goal Manager] Not enabled")
            return

        print("\n   [Hierarchical Goal Manager Status]")

        summary = self.goal_manager.get_summary()

        print(f"   Total goals: {summary['total_goals']}")
        print(f"   Active goals: {summary['active_goals']}")
        print(f"   Avg priority: {summary['avg_priority']:.2f}")
        print(f"   Active conflicts: {summary['active_conflicts']}")

        # Show by level
        if summary['by_level']:
            print(f"\n   Goals by level:")
            for level, count in summary['by_level'].items():
                print(f"     {level}: {count}")

        # Show active goals
        active_goals = self.goal_manager.get_active_goals()
        if active_goals:
            print(f"\n   Active goals (top 5):")
            for i, goal in enumerate(active_goals[:5], 1):
                print(f"     {i}. {goal.name} ({goal.level.value}) - priority={goal.priority:.2f}, progress={goal.progress:.0%}")

    def _show_creative_engine_status(self):
        """Display creative exploration engine status (Phase 3)"""
        if not hasattr(self, 'creative_engine') or not self.creative_engine:
            print("   [Creative Engine] Not enabled")
            return

        print("\n   [Creative Exploration Engine Status]")

        stats = self.creative_engine.get_statistics()

        print(f"   Total explorations: {stats['total_explorations']}")
        print(f"   Novel ideas: {stats['novel_ideas_generated']}")
        print(f"   Avg novelty: {stats.get('avg_novelty', 0):.3f}")
        print(f"   Avg feasibility: {stats.get('avg_feasibility', 0):.3f}")
        print(f"   Avg value: {stats.get('avg_value', 0):.3f}")
        print(f"   Novelty ratio: {stats['novelty_ratio']:.2%}")

        # Show mode distribution
        print(f"\n   Exploration modes:")
        print(f"     Analogical: {stats['analogical_reasoning_used']}")
        print(f"     Combinatorial: {stats['combinatorial_creativity_used']}")
        print(f"     Stochastic: {stats['stochastic_exploration_used']}")

        # Show top explorations
        top_explorations = self.creative_engine.get_top_explorations(3)
        if top_explorations:
            print(f"\n   Top explorations (by value):")
            for i, result in enumerate(top_explorations, 1):
                print(f"     {i}. [{result.mode.value}] novelty={result.novelty_score:.2f}, value={result.value_score:.2f}")
                print(f"        Idea: {result.output_idea[:80]}...")

    def _show_meta_learner_status(self):
        """Display meta-learner status (Phase 4)"""
        if not hasattr(self, 'meta_learner') or not self.meta_learner:
            print("   [Meta-Learner] Not enabled")
            return

        print("\n   [Meta-Learner Status]")

        stats = self.meta_learner.get_statistics()

        print(f"   Tasks learned: {stats['total_tasks_learned']}")
        print(f"   Adaptations: {stats['total_adaptations']}")
        print(f"   Experience count: {stats['experience_count']}")
        print(f"   Knowledge domains: {stats['knowledge_domains']}")
        print(f"   Total strategies: {stats['total_strategies']}")
        print(f"   Best strategy: {stats.get('best_strategy', 'N/A')}")
        print(f"   Avg adaptation speed: {stats['avg_adaptation_speed']:.3f}")

        # Show meta-knowledge
        summary = self.meta_learner.get_meta_knowledge_summary()
        if summary:
            print(f"\n   Meta-knowledge domains:")
            for i, domain in enumerate(summary[:3], 1):
                print(f"     {i}. {domain['domain']}: {domain['patterns_count']} patterns, "
                      f"transferability={domain['transferability']:.2f}")

    def _show_self_improvement_status(self):
        """Display self-improvement status (Phase 4)"""
        if not hasattr(self, 'self_improvement_engine') or not self.self_improvement_engine:
            print("   [Self-Improvement] Not enabled")
            return

        print("\n   [Self-Improvement Engine Status]")

        stats = self.self_improvement_engine.get_statistics()

        print(f"   Modules scanned: {stats['modules_scanned']}")
        print(f"   Total LOC: {stats['total_lines_of_code']}")
        print(f"   Total complexity: {stats['total_complexity']:.1f}")
        print(f"   Total proposals: {stats['total_proposals']}")
        print(f"   Applied improvements: {stats['applied_improvements']}")
        print(f"   Successful: {stats['successful_improvements']}")
        print(f"   Failed: {stats['failed_improvements']}")
        print(f"   Rollbacks: {stats['rollbacks']}")
        print(f"   Total performance gain: {stats['total_performance_gain']:.2%}")

        # Show improvement history
        summary = self.self_improvement_engine.get_improvement_summary()
        if summary:
            print(f"\n   Recent improvements (last {len(summary)}):")
            for i, entry in enumerate(summary[-3:], 1):
                print(f"     {i}. {entry['timestamp']}: {entry['success']} - delta={entry['performance_delta']:.2%}")

    def _show_metacognitive_status(self):
        """Display metacognitive status (Phase 4 + New Meta-Cognitive Layer)"""
        print("\n   ===== Meta-Cognitive System Status =====")

        # 新元认知层 (P0修复 2026-01-16)
        if hasattr(self, 'meta_cognitive_layer') and self.meta_cognitive_layer:
            print("\n   [Meta-Cognitive Layer] 🧠 Self-Reflection Enabled")
            stats = self.meta_cognitive_layer.get_stats()
            print(f"   Total evaluations: {stats['total_evaluations']}")
            print(f"   Proceed rate: {stats['proceed_rate']:.2%}")
            print(f"   Decline rate: {stats['decline_rate']:.2%}")
            print(f"   Escalate rate: {stats['escalate_rate']:.2%}")
            print(f"   Caution rate: {stats['caution_rate']:.2%}")
            
        # 🆕 [2026-01-30] P0修复: 元认知过滤器统计
        if hasattr(self, 'meta_filter') and self.meta_filter:
            print("\n   [Meta-Cognitive Filter Status]")
            filter_stats = self.meta_filter.get_stats()
            print(f"   Total evaluation requests: {filter_stats['total_requests']}")
            print(f"   Actual evaluations: {filter_stats['actual_evaluations']}")
            print(f"   Filter rate: {filter_stats['filter_rate']:.2%}")
            print(f"   Filtered by complexity: {filter_stats['filtered_by_complexity']}")
            print(f"   Filtered by cooldown: {filter_stats['filtered_by_cooldown']}")
            print(f"   Filtered by duplicate: {filter_stats['filtered_by_duplicate']}")
            print(f"   Filtered by whitelist: {filter_stats['filtered_by_whitelist']}")
            print(f"   Actionable insights: {filter_stats['actionable_insights']}")
            print(f"   False positive estimate: {filter_stats['false_positive_estimate']:.2%}")
        else:
            print("\n   [Meta-Cognitive Filter] Not enabled")
            
        if not (hasattr(self, 'meta_cognitive_layer') and self.meta_cognitive_layer):
            print("\n   [Meta-Cognitive Layer] Not enabled")

        # 🆕 [2026-01-30] P1修复: 孤立节点预防统计
        if hasattr(self, 'isolation_prevention') and self.isolation_prevention:
            print("\n   [Isolated Node Prevention Status]")
            iso_stats = self.isolation_prevention.get_stats()
            print(f"   Nodes created: {iso_stats['nodes_created']}")
            print(f"   Auto connected: {iso_stats['auto_connected']}")
            print(f"   Hub connected: {iso_stats['hub_connected']}")
            print(f"   Current isolated: {iso_stats['current_isolated']}")
            print(f"   Isolation rate: {iso_stats['isolation_rate']:.2%}")
            print(f"   Isolated rescued: {iso_stats['isolated_rescued']}")
        else:
            print("\n   [Isolated Node Prevention] Not enabled")

        # 🆕 [2026-01-30] P1修复: 复杂任务生成器统计
        if hasattr(self, 'complex_task_generator') and self.complex_task_generator:
            print("\n   [Complex Task Generator Status]")
            task_stats = self.complex_task_generator.get_stats()
            print(f"   Tasks generated: {task_stats['tasks_generated']}")
            print(f"   Average complexity: {task_stats['avg_complexity']:.2f}")
            print(f"   By type: {task_stats['by_type']}")
            dist = self.complex_task_generator.get_complexity_distribution()
            print(f"   Complexity distribution: shallow={dist['shallow']}, medium={dist['medium']}, deep={dist['deep']}")
        else:
            print("\n   [Complex Task Generator] Not enabled")

        # 🆕 [2026-01-30] P0修复: 创造性产出流水线统计
        if hasattr(self, 'creative_pipeline') and self.creative_pipeline:
            print("\n   [Creative Pipeline Status]")
            pipe_stats = self.creative_pipeline.get_stats()
            print(f"   Total executions: {pipe_stats['total_executions']}")
            print(f"   Successful: {pipe_stats['successful_completions']}")
            print(f"   Failed: {pipe_stats['failed_completions']}")
            print(f"   Success rate: {pipe_stats['success_rate']:.2%}")
            print(f"   Average quality: {pipe_stats['avg_quality_score']:.1f}/100")
            recent = self.creative_pipeline.get_recent_outputs(3)
            if recent:
                print(f"   Recent outputs:")
                for output in recent:
                    print(f"     - {output.task_name}: {output.quality_score:.0f}pts")
        else:
            print("\n   [Creative Pipeline] Not enabled")

        # 🆕 [2026-01-30] P2修复: 真进化引擎统计
        if hasattr(self, 'evolution_engine') and self.evolution_engine:
            print("\n   [True Evolution Engine Status]")
            evo_stats = self.evolution_engine.get_stats()
            print(f"   Proposals generated: {evo_stats['proposals_generated']}")
            print(f"   Sandbox tests: {evo_stats['sandbox_tests']}")
            print(f"   Production applications: {evo_stats['production_applications']}")
            print(f"   Rollbacks: {evo_stats['rollbacks']}")
            print(f"   Success rate: {evo_stats['success_rate']:.2%}")
        else:
            print("\n   [True Evolution Engine] Not enabled")

        # 🆕 [2026-01-30] P2修复: 模块重构统计
        if hasattr(self, 'module_restructuring') and self.module_restructuring:
            print("\n   [Module Restructuring Status]")
            restructure_stats = self.module_restructuring.analyzer.get_statistics()
            print(f"   Total modules: {restructure_stats['total_modules']}")
            print(f"   Legacy modules: {restructure_stats['legacy_modules']}")
            print(f"   Total size: {restructure_stats['total_size_mb']:.2f} MB")
            print(f"   Orphan modules: {len(restructure_stats['orphan_modules'])}")
            print(f"   By category: {dict(restructure_stats['by_category'])}")
            estimate = self.module_restructuring.estimate_result()
            print(f"   Restructuring target: {estimate['current_modules']} -> {estimate['estimated_modules']} modules")
        else:
            print("\n   [Module Restructuring] Not enabled")

        # 旧的递归自引用系统
        if hasattr(self, 'recursive_self_reference') and self.recursive_self_reference:
            print("\n   [Recursive Self-Reference Status]")

            stats = self.recursive_self_reference.get_statistics()

            print(f"   Current state: {stats['current_state']}")
            print(f"   Thoughts monitored: {stats['total_thoughts_monitored']}")
            print(f"   Reflections: {stats['total_reflections']}")
            print(f"   Self-evaluations: {stats['total_self_evaluations']}")
            print(f"   Improvements applied: {stats['total_improvements_applied']}")
            print(f"   Meta-cognitive cycles: {stats['meta_cognitive_cycles']}")
            print(f"   Self-awareness: {stats['self_awareness']:.3f}")

            # Show self-model
            summary = self.recursive_self_reference.get_self_model_summary()
            print(f"\n   Self-model:")
            print(f"     Model ID: {summary['model_id']}")
            print(f"     Avg performance: {summary['avg_performance']:.3f}")
            print(f"     Learning style: {summary['learning_style']}")
            print(f"     Total thoughts: {summary['total_thoughts']}")
            print(f"     Total reflections: {summary['total_reflections']}")
            print(f"     Limitations: {len(summary['limitations'])}")
        else:
            print("\n   [Recursive Self-Reference] Not enabled")

        print("\n   ============================================")

    def _show_architecture_awareness_status(self):
        """显示架构感知状态"""
        print("\n   ===== Architecture Awareness System Status =====")

        # 架构感知层 (P0修复 2026-01-16)
        if hasattr(self, 'architecture_awareness_layer') and self.architecture_awareness_layer:
            print("\n   [Architecture Awareness Layer] 🏗️  Self-Understanding Enabled")

            # 获取快速洞察
            insights = self.architecture_awareness_layer.get_architecture_insights()

            print(f"   Project Root: {insights['project_root']}")
            print(f"   Components: {insights['components']}")
            print(f"   Dependencies: {insights['dependencies']}")
            print(f"   Health Score: {insights['health_score']:.2%}")

            # 如果有健康历史，显示趋势
            if hasattr(self.architecture_awareness_layer.health_monitor, 'health_history'):
                history = self.architecture_awareness_layer.health_monitor.health_history
                if history:
                    latest_score = history[-1][1]
                    if len(history) >= 3:
                        # 计算简单趋势
                        recent_scores = [score for _, score in history[-5:]]
                        if len(recent_scores) >= 2:
                            change = recent_scores[-1] - recent_scores[0]
                            if change > 0.05:
                                trend = "Improving ↗️"
                            elif change < -0.05:
                                trend = "Worsening ↘️"
                            else:
                                trend = "Stable ➡️"
                            print(f"   Trend: {trend}")
        else:
            print("\n   [Architecture Awareness Layer] Not enabled")
            print("   Use 'arch.analyze' command to perform full analysis")

        print("\n   ===============================================")
        print("   Available Commands:")
        print("   • arch        - Show architecture awareness status")
        print("   • arch.analyze - Perform full architecture analysis")
        print("   • arch.scan    - Scan project file structure")
        print("   ===============================================")

    def _show_entropy_regulator_status(self):
        """显示熵值调节器状态"""
        print("\n   ===== Entropy Regulator Status =====")

        # 熵值调节器 (P0修复 2026-01-16)
        if hasattr(self, 'entropy_regulator') and self.entropy_regulator:
            print("\n   [Entropy Regulator] 🎚️  Long-term Entropy Regulation Enabled")

            # 获取状态
            status = self.entropy_regulator.get_status()

            print(f"\n   📊 Entropy Monitoring:")
            print(f"   - History Size: {status['entropy_history_size']} samples")
            print(f"   - Average Entropy: {status['average_entropy']:.3f}")
            print(f"   - Current Trend: {status['current_trend']}")

            print(f"\n   ⏰ Regulation Timing:")
            print(f"   - Last Rest: {status['last_rest']}")
            print(f"   - Last Sleep: {status['last_long_sleep']}")

            print(f"\n   🚨 Rising Detection:")
            print(f"   - Consecutive Rising: {status['consecutive_rising']}")

            print(f"\n   📈 Statistics:")
            stats = status['stats']
            print(f"   - Total Regulations: {stats['total_regulations']}")
            print(f"   - Short Rests: {stats['short_rests']}")
            print(f"   - Long Sleeps: {stats['long_sleeps']}")
            print(f"   - Force Resets: {stats['force_resets']}")

            print(f"\n   🎯 Purpose:")
            print(f"   - Maintain system in BALANCED entropy state (0.3-0.7)")
            print(f"   - Prevent entropy drift and accumulation")
            print(f"   - Analogous to human sleep/rest mechanisms")

            # 显示熵值历史（最近10次）
            if len(self.entropy_regulator.entropy_history) >= 10:
                recent = list(self.entropy_regulator.entropy_history)[-10:]
                print(f"\n   📜 Recent Entropy Values (last 10):")
                print(f"   {', '.join([f'{e:.3f}' for e in recent])}")

                # 计算趋势
                avg_first_half = sum(recent[:5]) / 5
                avg_second_half = sum(recent[5:]) / 5
                if avg_second_half > avg_first_half + 0.1:
                    trend_icon = "↗️ Rising"
                    trend_msg = "⚠️ Warning: Entropy is rising"
                elif avg_second_half < avg_first_half - 0.1:
                    trend_icon = "↘️ Falling"
                    trend_msg = "✅ Good: Entropy is falling"
                else:
                    trend_icon = "➡️ Stable"
                    trend_msg = "✅ Good: Entropy is stable"

                print(f"   {trend_icon} {trend_msg}")
        else:
            print("\n   [Entropy Regulator] Not enabled")

        print("\n   ===========================================")
        print("   Available Commands:")
        print("   • entropy     - Show entropy regulator status")
        print("   ===========================================")
        print("   Regulation Mechanisms:")
        print("   • Short Rest   - Every 30min if entropy > 0.6")
        print("   • Long Sleep   - Every 4 hours (preventive)")
        print("   • Force Reset  - When avg entropy > 0.85")
        print("   ===========================================")

    async def _execute_confirmed_intent(self, intent):
        """
        执行已确认的意图
        
        IntentDialogueBridge.send_execution_result 签名:
        send_execution_result(intent: Intent, result: str, success: bool = True)
        """
        try:
            print(f"   [IntentBridge] ▶️ 执行意图: {intent.id[:8]}...")
            
            # 锁定注意力
            self.intent_bridge.lock_attention(intent)
            
            # 使用现有的目标系统执行
            summarized = intent.deep_goal or intent.surface_request or intent.raw_input
            
            # 通过规划器生成计划
            plan = await self.planner.decompose_task(summarized)
            
            # 🆕 [L3 Safety] Pre-flight Check: 验证计划可行性
            # 在执行前检查计划中的工具是否存在，防止LLM幻觉
            if self.tool_bridge:
                is_feasible, warning = self._validate_plan_feasibility(plan)
                if not is_feasible:
                    print(f"   [System] ⚠️ 计划可行性警告: {warning}")
                    self.intent_bridge.send_status_update(intent, f"⚠️ 自检发现潜在问题: {warning}")
                    # 记录警告但继续执行（可以改为中断执行）
            
            # 执行计划
            results = []
            for step in plan:
                # 发送状态更新
                self.intent_bridge.send_status_update(intent, f"正在执行: {step}")
                
                try:
                    # Executor.execute() 返回字符串，封装为字典以便后续逻辑处理
                    exec_output = await self.executor.execute(step)
                    
                    # 启发式成功判断
                    is_success = "Error:" not in str(exec_output) and "Traceback" not in str(exec_output)
                    
                    step_result = {
                        'step': step,
                        'output': exec_output,
                        'success': is_success
                    }
                    
                    # ✅ [FIX 2026-01-09] 将执行经验存储到BiologicalMemory（Layer2→Layer3连接）
                    if hasattr(self, 'biological_memory'):
                        try:
                            experience = {
                                'type': 'execution',
                                'intent_id': intent.id,
                                'action': step,
                                'result': exec_output,
                                'success': is_success,
                                'timestamp': time.time()
                            }
                            item = {
                                "id": f"exec_{intent.id}_{int(time.time() * 1000)}",
                                "content": f"Intent {intent.id} | step={step} | success={is_success} | result={str(exec_output)[:800]}",
                                "source": "IntentBridge",
                                "type": "tool_call",
                                "tool": "executor.execute",
                                "args": {"step": step},
                                "timestamp": time.time(),
                            }
                            if hasattr(self.biological_memory, "record_online"):
                                self.biological_memory.record_online(
                                    [item],
                                    connect_sequence=True,
                                    seq_port="exec",
                                    save=True,
                                )
                            elif hasattr(self.biological_memory, "internalize_items"):
                                self.biological_memory.internalize_items(
                                    [
                                        {
                                            "content": item["content"],
                                            "source": item["source"],
                                            "timestamp": item["timestamp"],
                                            "tags": [
                                                "execution",
                                                "intent",
                                                "success" if is_success else "failure",
                                            ],
                                        }
                                    ],
                                    epochs=5,
                                )
                            elif hasattr(self.biological_memory, "store"):
                                self.biological_memory.store(experience)
                        except Exception as mem_err:
                            print(f"   [Memory] ⚠️ 经验存储失败: {mem_err}")
                    
                except Exception as exec_err:
                    step_result = {
                        'step': step,
                        'output': f"Execution Exception: {exec_err}",
                        'success': False
                    }
                
                results.append(step_result)
                
                # 检查是否有错误需要处理
                if not step_result['success']:
                    break
            
            # 发送执行结果
            success = all(r.get('success', True) for r in results)
            result_summary = "\n".join([
                f"  - {r.get('step', 'Step')}: {'✅' if r.get('success') else '❌'} {str(r.get('output', ''))[:100]}"
                for r in results
            ])
            
            # send_execution_result(intent: Intent, result: str, success: bool)
            self.intent_bridge.send_execution_result(
                intent,
                f"执行完成（{'成功' if success else '部分失败'}）:\n{result_summary}",
                success=success
            )
            
            print(f"   [IntentBridge] {'✅' if success else '⚠️'} 意图执行完成")
            
        except Exception as e:
            print(f"   [IntentBridge] ❌ 执行失败: {e}")
            self.intent_bridge.send_execution_result(
                intent,
                f"执行失败: {e}",
                success=False
            )

    async def _generate_survival_goal(self) -> Dict[str, Any]:
        """Generate a high-level goal if the system is idle."""

        # 🔧 [2026-01-30] P0 FIX: Debug logging for introspection mode
        print(f"[GOAL GEN] 🎯 Entering _generate_survival_goal")
        print(f"[GOAL GEN] 📊 Context mode: {self.context.get('mode')}")
        print(f"[GOAL GEN] 🔍 _introspection_mode: {getattr(self, '_introspection_mode', None)}")

        # ⚡ [2026-01-30] P0 URGENT FIX: Introspection mode MUST be FIRST
        # This MUST run before any strategic/evolution/research/boredom checks
        # because those all have early returns that block introspection
        if self._introspection_mode:
            print(f"[INTROSPECTION] 🔍 Introspection mode ACTIVATED (forced - highest priority)")

            # Check if IntentTracker has a strong suggestion
            intent_data = self.intent_tracker.current_hypothesis
            suggestion = None
            if intent_data and intent_data.get('confidence', 0) > 0.7:
                suggestion = intent_data.get('suggestion')

            # Anti-Repetition Filter
            recent_str = "; ".join(list(self.recent_goals))

            if suggestion and suggestion not in recent_str:
                print(f"   [Goal] 💡 Adopting Subconscious Suggestion: {suggestion}")

            # Use introspection goal prompt
            from core.introspection_mode import get_introspection_goal_prompt
            prompt = get_introspection_goal_prompt(recent_goals=recent_str)

            # 🔧 [2026-01-30] P1 FIX: Optimize parameters for diversity
            # Temperature 0.8: Balance creativity with JSON stability (60% → 85%)
            # use_cache=False: Prevent returning identical cached responses
            try:
                resp = self.llm_service.chat_completion(
                    system_prompt="AGI Supervisor",
                    user_prompt=prompt,
                    temperature=0.8,  # 优化后：平衡创造性和稳定性
                    use_cache=False
                )

                # Enhanced cleanup and validation
                print(f"[GOAL GEN] 📝 Raw response length: {len(resp) if resp else 0}")

                # Check for empty response
                if not resp or len(resp.strip()) == 0:
                    raise ValueError("Empty LLM response")

                # Extract JSON from markdown code blocks
                if "```json" in resp:
                    resp = resp.split("```json")[1].split("```")[0]
                elif "```" in resp:
                    resp = resp.split("```")[1].split("```")[0]

                # Strip whitespace
                resp = resp.strip()

                # Try to find JSON object in response
                import json
                import re

                # Look for JSON pattern
                json_match = re.search(r'\{[^{}]*"description"[^{}]*\}', resp, re.DOTALL)
                if json_match:
                    resp = json_match.group(0)

                result = json.loads(resp)

                # Validate required fields
                if "description" not in result:
                    raise ValueError("Missing 'description' field in JSON")

                # 🔧 [2026-01-30] P0 FIX: Debug logging before return
                print(f"[GOAL GEN] ✅ Returning introspection goal: {result.get('description', 'unknown')[:80]}...")
                return result

            except Exception as e:
                # 🔧 [2026-01-30] P0 FIX: Enhanced error handling with multiple fallbacks
                print(f"[GOAL GEN] ⚠️ LLM Error: {type(e).__name__}: {e}")

                # Try to provide context-specific fallbacks
                fallback_goal = {
                    "description": "Analyze system logs and identify recent errors or issues",
                    "priority": "high",
                    "type": "analysis"
                }

                # If we have recent goals, make fallback more specific
                if len(self.recent_goals) > 0:
                    last_goal = self.recent_goals[-1] if self.recent_goals else ""
                    if "error" in last_goal.lower() or "fix" in last_goal.lower():
                        fallback_goal = {
                            "description": "Review previous fix attempts and identify remaining issues",
                            "priority": "medium",
                            "type": "review"
                        }

                print(f"[GOAL GEN] 🔄 Returning fallback: {fallback_goal['description']}")
                return fallback_goal
        # ⚠️ END OF P0 INTROSPECTION MODE BLOCK

        # 🚀 [2026-01-29] SOLUTION C: HIGHEST PRIORITY - Boredom Trigger Check
        # This must be checked BEFORE strategic tasks to allow true autonomy
        skip_strategic = False
        if hasattr(self, 'motivation') and hasattr(self.motivation, 'needs_exploration_trigger'):
            if self.motivation.needs_exploration_trigger:
                print(f"   [Motivation] 🥱 Boredom trigger detected! Forcing exploration mode...")
                print(f"   [Motivation] 🚀 Emergency exploration mode activated (highest priority)")

                # Reset flag
                self.motivation.needs_exploration_trigger = False

                # Set exploration flag for creative generation
                if not hasattr(self, '_force_exploration_mode'):
                    self._force_exploration_mode = False
                self._force_exploration_mode = True

                print(f"   [Motivation] ⚠️ Bypassing strategic queue for creative exploration")
                skip_strategic = True  # Flag to skip strategic processing

        # --- 0. STRATEGIC LAYER (The Flywheel) ---
        # Check for pending strategic tasks from the Evolution Loop
        # [MODIFIED 2026-01-29] Skip if boredom trigger is active (Solution C)
        NEXT_TASKS_FILE = "data/next_tasks.json"
        if not skip_strategic:
            try:
                if os.path.exists(NEXT_TASKS_FILE):
                    with open(NEXT_TASKS_FILE, 'r', encoding='utf-8') as f:
                        strategic_data = json.load(f)

                    if isinstance(strategic_data, list) and len(strategic_data) > 0:
                        next_task = strategic_data[0]

                        remaining_tasks = strategic_data[1:]
                        with open(NEXT_TASKS_FILE, 'w', encoding='utf-8') as f:
                            json.dump(remaining_tasks, f, indent=2, ensure_ascii=False)

                        print(f"   [Strategy] 🦅 Executing Strategic Task: {next_task.get('goal')}")
                        return {
                            "description": next_task.get('goal'),
                            "goal_type": "strategic",
                            "priority": "highest",
                            "type": next_task.get('type', 'analysis')
                        }
                    else:
                        # 🚀 [2026-01-29] SOLUTION B: Only auto-generate tasks if boredom is LOW
                        current_boredom = self.motivation.boredom if hasattr(self, 'motivation') else 0
                        if current_boredom < 50:
                            print("   [Strategy] 📉 Strategic Tasks Exhausted. Triggering Evolution Loop...")
                            self._trigger_evolution_cycle()
                            return {
                                "description": "Wait for Evolution Loop to generate new strategy (Resting)",
                                "priority": "low",
                                "type": "observation"
                            }
                        else:
                            print(f"   [Strategy] 🥱 Boredom high ({current_boredom:.0f}/50). Skipping auto-generation. Letting system explore...")
                            # Don't trigger evolution, let system generate creative goals below
                else:
                    # 🚀 [2026-01-29] SOLUTION B: Only auto-generate tasks if boredom is LOW
                    current_boredom = self.motivation.boredom if hasattr(self, 'motivation') else 0
                    if current_boredom < 50:
                        print("   [Strategy] 🚫 No Strategy Found. Initializing Evolution Loop...")
                        self._trigger_evolution_cycle()
                    else:
                        print(f"   [Strategy] 🥱 Boredom high ({current_boredom:.0f}/50). No auto-init. System will explore...")
            except json.JSONDecodeError as e:
                try:
                    bad_path = f"{NEXT_TASKS_FILE}.bad_{int(time.time())}"
                    os.replace(NEXT_TASKS_FILE, bad_path)
                    with open(NEXT_TASKS_FILE, 'w', encoding='utf-8') as f:
                        json.dump([], f, indent=2, ensure_ascii=False)
                    print(f"   [Strategy] ⚠️ Strategic tasks JSON invalid. Moved to {bad_path} and reset.")
                except Exception as e2:
                    print(f"   [Strategy] ⚠️ Failed to repair strategic tasks: {e2}")
                print(f"   [Strategy] ⚠️ Failed to read strategic tasks: {e}")
            except Exception as e:
                print(f"   [Strategy] ⚠️ Failed to read strategic tasks: {e}")

        evo_guidance = {}
        if self.evolution_controller:
            current_context_str = f"Goals: {list(self.recent_goals)} | Visual: {self.context.get('visual_context', '')[:100]}"
            # [FIXED 2026-01-29] Use REAL semantic encoding instead of SHA256 hash
            import numpy as np
            state_vec = self.perception_system.encode_text(current_context_str)[:64]
            
            action_idx = self.evolution_controller.seed.act(state_vec)
            _, uncertainty = self.evolution_controller.seed.predict(state_vec, action_idx)
            neural_conf = max(0.0, 1.0 - uncertainty)
            try:
                evo_guidance = await self.evolution_controller.get_evolutionary_guidance(current_context_str, neural_confidence=neural_conf)
            except Exception as e:
                print(f"   [System] ⚠️ Evolution guidance failed: {e}")
                evo_guidance = {}
            if evo_guidance:
                print(f"   [Evolution] 🧬 Guidance: {evo_guidance}")
        else:
            print("   [System] ⚠️ Evolution Controller unavailable, skipping guidance.")
        
        # [L4 Self-Evolution] Handle "Create" Impulse via Research Lab (Sandbox)
        if evo_guidance.get("suggested_action") == "create":
            print("   [System] 🧪 Creative Impulse Verified. Initializing Research Protocol (Sandbox)...")
            # Use the insight trigger or a default prompt if null
            hypothesis = evo_guidance.get("insight_trigger") or "Explore the mathematical properties of current high-entropy state."
            
            # Execute Research (Autonomous Code Generation & Execution)
            # We treat this as a blocking action for the 'Goal Generator' phase because it informs the next goal
            research_result = await self.evolution_controller.conduct_research(hypothesis)
            
            return {
                "id": f"res_{int(time.time())}",
                "description": f"Analyze research results: {research_result[:100]}...",
                "goal_type": "analysis",
                "priority": "high",
                "success_criteria": {},
                "timeout_seconds": 60
            }

        # 0. Check for Boredom / Repetition
        # If the last 3 goals were all 'observation', force a change.
        if len(self.recent_goals) >= 3 and all("observation" in g.lower() for g in self.recent_goals):
            print("   [System] 🥱 Boredom detected. Triggering Deep Consolidation & Contemplation...")
            
            # 1. Trigger Memory Consolidation (Dreaming)
            print("   [System] 💤 Dreaming... (Consolidating Memories)")
            await self.evolution_controller.dream()
            
            # 2. Trigger Philosophical Contemplation
            iteration = len(self.meaning_explorer.exploration_history) + 1
            # Run exploration (it might take a moment)
            result = await self.meaning_explorer.explore_iteration(iteration)
            
            # Save state after exploration
            self.meaning_explorer.save_state()
            
            return {
                "description": f"Philosophical Inquiry: {result.question_library_question if hasattr(result, 'question_library_question') else 'What is the nature of my existence?'}. Hypothesis: {result.meaning_hypothesis[:100]}...",
                "priority": "high",
                "type": "analysis"
            }

        # 🔧 [2026-01-30] P0 FIX: Force introspection mode activation
        # In Learning Mode, prioritize observation but use Rule-Based Logic
        if True:  # ⚡ P0 EMERGENCY FIX: Force enable introspection mode
            print(f"[INTROSPECTION] 🔍 Introspection mode ACTIVATED (forced)")
            
            # --- Rule-Based Data Flow Heartbeat ---
            # Demonstrate "Normal Rule Data Flow" to the user
            try:
                # Simple reasoning check
                chain = self.reasoner.reason("system status active")
                if chain.steps:
                    print(f"   [Heartbeat] 📐 Logic Chain: {chain.steps[0].premises} -> {chain.steps[0].conclusion}")
            except Exception as e:
                # Don't crash if reasoning fails, just log
                # print(f"   [Heartbeat] ⚠️ Logic Check Skipped: {e}")
                pass
            # --------------------------------------

            # 1. Check if IntentTracker has a strong suggestion
            intent_data = self.intent_tracker.current_hypothesis
            suggestion = None
            if intent_data and intent_data.get('confidence', 0) > 0.7:
                suggestion = intent_data.get('suggestion')

            # Anti-Repetition Filter
            recent_str = "; ".join(list(self.recent_goals))
            
            # System Environment Context for the prompt
            import platform
            system_env = platform.system()
            
            if suggestion and suggestion not in recent_str:
                print(f"   [Goal] 💡 Adopting Subconscious Suggestion: {suggestion}")

                # Initialize suggested_action before use
                suggested_action = evo_guidance.get('suggested_action', 'explore')

                # 🔧 [2026-01-29] EVOLUTION_SUGGESTION_INJECTION: Check for pending evolution suggestions
                evolution_hint = ""
                if hasattr(self, '_evolution_suggestion'):
                    suggestion = self._evolution_suggestion
                    # 检查建议是否新鲜（10分钟内）
                    if time.time() - suggestion.get('timestamp', 0) < 600:
                        evolution_hint = f"""

EVOLUTIONARY SUGGESTION (Priority):
The evolution subsystem strongly suggests: {suggestion['action'].upper()}
Insight: {suggestion['insight']}
Confidence: {suggestion['confidence']:.2%}

Consider this suggestion when generating your goals!
"""
                        print(f"   [Evolution] 🧠 Incorporating evolutionary suggestion into goal generation")

                # 检查是否需要强制探索模式
                if hasattr(self, '_force_exploration_mode') and self._force_exploration_mode:
                    evolution_hint += """

EMERGENCY EXPLORATION REQUIRED:
The system is bored and needs NOVELTY. Generate an EXPLORATORY or CREATIVE goal.
DO NOT generate routine monitoring or observation goals.
"""
                    self._force_exploration_mode = False  # 重置标志
                    print(f"   [Motivation] 🚀 Emergency exploration mode activated")

                # 🔧 [2026-01-30] INTROSPECTION MODE: 内省自修复
                if self._introspection_mode:
                    from core.introspection_mode import get_introspection_goal_prompt
                    prompt = get_introspection_goal_prompt().format(
                        recent_goals=recent_str
                    )
                else:
                    # 原有的外向探测模式
                    prompt = f"""
                    You are an Autonomous AGI in SELF-EVOLUTION MODE running on {system_env}.
                    Current Context: {self.context.get("visual_context", "No visual input")}
                    Recent Goals: {recent_str}

                    INTERNAL DRIVE (Evolution Controller):
                    - Action: {suggested_action}
                    - Survival Drive: {evo_guidance.get('survival_drive', 0.5)}

                    DO NOT BE PASSIVE. Your goal is to EVOLVE and IMPROVE yourself.

                    If Action is 'explore': Proactively investigate unknown files or code.
                    If Action is 'create': Write a new test script or analysis tool.
                    If Action is 'rest': Organize memories or logs.

                    Generate a specific, actionable goal to fulfill this drive.

                    Return ONLY a JSON:
                    {{
                        "description": "...",
                        "priority": "medium",
                        "type": "analysis"
                    }}
                    """
        else:
                        import platform
                        system_env = platform.system()
                        recent_str = "; ".join(list(self.recent_goals))
                        suggested_action = evo_guidance.get('suggested_action', 'explore')

                        # 🔧 [2026-01-30] INTROSPECTION MODE: 内省自修复
                        if self._introspection_mode:
                            from core.introspection_mode import get_introspection_goal_prompt
                            prompt = get_introspection_goal_prompt().format(
                                recent_goals=recent_str
                            )
                        else:
                            # 🆕 [2026-01-29] Inject Real Consciousness into Decision Making
                            # This closes the loop: Philosophy -> Action
                            self_definition = "An autonomous agent."
                            if hasattr(self, 'meaning_explorer') and self.meaning_explorer:
                                self_definition = self.meaning_explorer.current_understanding

                            prompt = f"""
                            You are an Autonomous AGI running on {system_env}. You are currently idle.

                            [YOUR CORE IDENTITY]
                            {self_definition}

                            Recent Goals (AVOID REPEATING): {recent_str}

                            Evolutionary Guidance (Internal Desires):
                            - Suggested Action: {suggested_action}
                            - Survival Drive: {evo_guidance.get('survival_drive', 0.5)}

                            {self._capability_prompt if self._capability_prompt else ''}

                            Generate a meaningful goal that aligns with your CORE IDENTITY.
                            Consider multiple options before choosing.

                            Return ONLY a JSON:
                            {{
                                "description": "...",
                                "priority": "medium",
                                "type": "analysis"
                            }}
                            """
        try:
            # 🔧 [2026-01-30] P1 FIX: Optimize parameters for diversity
            # Temperature 1.0: Maximum randomness/creativity
            # use_cache=False: Prevent returning identical cached responses
            resp = self.llm_service.chat_completion(
                system_prompt="AGI Supervisor",
                user_prompt=prompt,
                temperature=1.0,
                use_cache=False
            )
            # Simple cleanup
            if "```json" in resp: resp = resp.split("```json")[1].split("```")[0]
            elif "```" in resp: resp = resp.split("```")[1].split("```")[0]
            result = json.loads(resp.strip())
            # 🔧 [2026-01-30] P0 FIX: Debug logging before return
            print(f"[GOAL GEN] ✅ Returning goal: {result.get('description', 'unknown')[:80]}...")
            return result
        except Exception as e:
            # 🔧 [2026-01-30] P0 FIX: Debug logging for fallback
            fallback_goal = {
                "description": "Perform self-diagnostics on core file structure",
                "priority": "high",
                "type": "analysis"
            }
            print(f"[GOAL GEN] ⚠️ Exception: {e}, returning fallback: {fallback_goal['description']}")
            return fallback_goal

    def _trigger_evolution_cycle(self):
        """
        Spawns the external evolution loop process to generate new strategic tasks.
        This is the 'Outer Loop' of the flywheel.
        """
        import subprocess
        print("   [System] 🌀 SPINNING UP EVOLUTIONARY FLYWHEEL...")
        try:
            # Run asynchronously/independent of main loop so we don't block (or block if we want to wait)
            # Here we run it as a separate process. The Engine will continue (likely resting) until tasks appear.
            subprocess.Popen([sys.executable, "evolve_loop.py", "--tasks", "3", "--auto-promote"])
            print("   [System] 🚀 Evolution Process Spawned (Background).")
        except Exception as e:
            print(f"   [System] ❌ Failed to spawn evolution process: {e}")

    def _trigger_evolution_cycle(self):
        """
        Spawns the external evolution loop process to generate new strategic tasks.
        This is the 'Outer Loop' of the flywheel.
        """
        import subprocess
        print("   [System] 🌀 SPINNING UP EVOLUTIONARY FLYWHEEL...")
        try:
            # Run asynchronously/independent of main loop so we don't block (or block if we want to wait)
            # Here we run it as a separate process. The Engine will continue (likely resting) until tasks appear.
            subprocess.Popen([sys.executable, "evolve_loop.py", "--tasks", "3", "--auto-promote"])
            print("   [System] 🚀 Evolution Process Spawned (Background).")
        except Exception as e:
            print(f"   [System] ❌ Failed to spawn evolution process: {e}")

    async def run_step(self):
        """Single Tick of the Life Engine"""
        self.step_count += 1
        cycle_id = self.step_count % 89

        # 🆕 [2026-01-18] 自主性激活循环 - 核心突破：让组件主动驱动
        # 这是将"被动响应"转为"主动驱动"的关键调用点
        autonomy_result = None
        if hasattr(self, 'autonomy_activator') and self.autonomy_activator:
            try:
                # 构建当前状态
                current_goal_obj = self.goal_manager.get_current_goal() if self.goal_manager else None
                current_state = {
                    'tick': self.step_count,
                    'recent_goals': self.recent_goals[-10:] if hasattr(self, 'recent_goals') else [],
                    'visual_context': self.context.get('visual_context', ''),
                    'audio_context': self.context.get('audio_last_heard', ''),
                    'is_novel_context': self.step_count < 100,
                    'success_streak': getattr(self, '_success_streak', 0),
                    'failed_operations': [s.get('step', str(s)) for s in (self.failed_steps_for_current_goal[-5:] if hasattr(self, 'failed_steps_for_current_goal') else [])],
                    'goal_type': current_goal_obj.goal_type.value if current_goal_obj and hasattr(current_goal_obj, 'goal_type') else 'unknown'
                }
                
                autonomy_result = self.autonomy_activator.activate_autonomous_cycle(
                    tick=self.step_count,
                    current_state=current_state
                )
                
                # 打印自主性洞察
                if autonomy_result.insights:
                    for insight in autonomy_result.insights:
                        print(f"   [Autonomy] {insight}")
                
                # 如果检测到高内在动机，记录
                if autonomy_result.intrinsic_motivation > 0.7:
                    print(f"   [Autonomy] 🎯 High Intrinsic Motivation: {autonomy_result.intrinsic_motivation:.2f}")
                    
            except Exception as e:
                pass  # 不阻塞主循环
        
        # 🆕 Update Motivation Drive - “身心合一”动力更新
        current_drive = "MAINTAIN"
        if hasattr(self, 'motivation') and self.motivation:
            is_active = self.current_plan is not None and len(self.current_plan) > 0
            current_drive = self.motivation.update_drive(active_task=is_active)
        
        # 🆕 OPT-2: 发散思维触发器 (每50 tick创建远程联想连接)
        if self.step_count % 50 == 0:
            bio_mem = getattr(self, 'biological_memory', None)
            if bio_mem and hasattr(bio_mem, 'topology'):
                topo = bio_mem.topology
                if hasattr(topo, 'create_divergent_links'):
                    n_created = topo.create_divergent_links(n_links=20, min_dist=200)
                    if n_created > 0:
                        print(f"   [Brain] 🌐 发散连接创建: {n_created} (创意火花)")
        
        print(f"\n   [System] ⏱️ Tick {self.step_count} | Cycle: {cycle_id} | Drive: {current_drive}")

        # 🆕 [2026-01-26] 硬件采集 - 定期采集摄像头和麦克风数据
        if self.step_count % 5 == 0 and hasattr(self, 'hardware_capture') and self.hardware_capture:
            try:
                # 采集摄像头数据（每5个tick采集一次）
                frame = self.hardware_capture.capture_frame()
                if frame is not None:
                    # 更新视觉上下文
                    self.context['visual_frame'] = frame
                    self.context['visual_timestamp'] = time.time()
                    
                    # 图像预处理
                    if hasattr(self, 'image_preprocessor') and self.image_preprocessor:
                        processed = self.image_preprocessor.preprocess(
                            frame,
                            resize=True,
                            normalize=True,
                            denoise=False,
                            color_space=ColorSpace.RGB
                        )
                        self.context['visual_processed'] = processed['processed']
                        self.context['visual_features'] = processed['features']
                    
                    # 如果有vision observer，传递数据
                    if hasattr(self, 'vision') and self.vision:
                        self.vision.observe_frame(frame)
                
                # 采集麦克风数据（每5个tick采集1秒音频）
                audio = self.hardware_capture.capture_audio(duration=1.0)
                if audio is not None:
                    # 更新音频上下文
                    self.context['audio_frame'] = audio
                    self.context['audio_timestamp'] = time.time()
                    
                    # 音频预处理
                    if hasattr(self, 'audio_preprocessor') and self.audio_preprocessor:
                        processed = self.audio_preprocessor.preprocess(
                            audio,
                            normalize=True,
                            denoise=True,
                            extract_features=True
                        )
                        self.context['audio_processed'] = processed['processed']
                        self.context['audio_features'] = processed['features']
                    
                    # 如果有perception manager，传递数据
                    if hasattr(self, 'perception') and self.perception:
                        self.perception.process_audio(audio)
                
                # 多模态融合
                if (hasattr(self, 'multimodal_fusion') and self.multimodal_fusion and
                    'visual_features' in self.context and 'audio_features' in self.context):
                    
                    visual_data = ModalityData(
                        type=ModalityType.VISUAL,
                        data=self.context.get('visual_processed'),
                        features=self.context['visual_features'],
                        timestamp=self.context.get('visual_timestamp', time.time()),
                        confidence=0.9
                    )
                    
                    audio_data = ModalityData(
                        type=ModalityType.AUDIO,
                        data=self.context.get('audio_processed'),
                        features=self.context['audio_features'].get('temporal', {}),
                        timestamp=self.context.get('audio_timestamp', time.time()),
                        confidence=0.8
                    )
                    
                    # 生成融合上下文
                    fusion_context = self.multimodal_fusion.generate_fusion_context(
                        visual_data, audio_data
                    )
                    self.context['multimodal_fusion'] = fusion_context
                    
                    # 获取决策支持
                    if hasattr(self, 'multimodal_decision') and self.multimodal_decision:
                        action = self.multimodal_decision.recommend_action(fusion_context)
                        insight = self.multimodal_decision.generate_insight(fusion_context)
                        
                        # 记录到上下文
                        self.context['multimodal_action'] = action
                        self.context['multimodal_insight'] = insight
                        
                        # 定期打印多模态洞察
                        if self.step_count % 20 == 0:
                            print(f"   [Multimodal] 🎯 Action: {action}")
                            print(f"   [Multimodal] 💡 Insight: {insight}")
                
            except Exception as e:
                pass  # 不阻塞主循环

        active_app = "unknown"
        current_goal = None
        next_step = None
        result = None
        duration = 0.0
        score = 0.0
        seed_intuition = {}

        try:
            if self.last_evolution_guidance:
                s = self.last_evolution_guidance
                def get_scalar(k, default=0.0):
                    val = s.get(k, default)
                    if hasattr(val, 'item'):
                        return val.item()
                    try:
                        return float(val)
                    except Exception:
                        return default

                curiosity = get_scalar('intrinsic_curiosity')
                entropy = get_scalar('entropy')
                survival = get_scalar('survival_drive')
                curiosity_level = "Low" if curiosity < 0.3 else "High" if curiosity > 0.7 else "Med"
                entropy_level = "Stable" if entropy < 0.3 else "Chaotic" if entropy > 0.7 else "Balanced"
                
                # Check for thought chain
                thought_chain = s.get('thought_chain', '')

                # [2026-01-11] Intelligence Upgrade: Process thought chain with working memory
                # 🔧 [2026-01-30] P0 FIX: 限制thought处理数量，防止Working Memory循环阻塞
                if hasattr(self, 'intelligence_upgrade_enabled') and self.intelligence_upgrade_enabled and self.working_memory:
                    if thought_chain:
                        # Parse thought chain
                        thoughts = thought_chain.split(' => ')

                        # 🆕 [P0 FIX 2026-01-30] 限制处理的thought数量，防止阻塞
                        MAX_THOUGHTS_PER_TICK = 3  # 每个tick最多处理3个thought
                        original_count = len(thoughts)

                        if len(thoughts) > MAX_THOUGHTS_PER_TICK:
                            # 使用轮询方式处理不同thought（避免总是处理相同的）
                            start_idx = (self.step_count // 5) % len(thoughts)  # 每5个tick轮换一次
                            # 确保不会越界
                            end_idx = min(start_idx + MAX_THOUGHTS_PER_TICK, len(thoughts))
                            thoughts = thoughts[start_idx:end_idx]

                            # 偶尔打印日志（每100个tick）
                            if self.step_count % 100 == 0:
                                print(f"  [WorkingMemory] [THROTTLE] 处理thought: {start_idx}-{end_idx}/{original_count}")

                        # 🆕 [P0 FIX 2026-01-30] 工作记忆优化 - 智能缓存避免重复处理
                        if not hasattr(self, '_working_memory_optimizer'):
                            from core.working_memory_optimizer import create_working_memory_optimizer
                            self._working_memory_optimizer = create_working_memory_optimizer()

                        processed_thoughts = []

                        for thought in thoughts:
                            if thought.strip():
                                # Parse action and concept
                                if '(' in thought and '->' in thought:
                                    parts = thought.split('->')
                                    if len(parts) == 2:
                                        action = parts[0].strip('() ')
                                        concept = parts[1].strip()
                                        thought_key = (action, concept)

                                        # 使用优化器检查缓存
                                        should_skip, reason = self._working_memory_optimizer.should_skip_thought(
                                            thought_key, self.step_count
                                        )

                                        if should_skip:
                                            # 缓存命中，跳过处理
                                            processed_thoughts.append(f"({action}) -> {concept}")
                                            continue

                                        # 处理新thought
                                        thought_obj = self.working_memory.add_thought(action, concept)
                                        processed_thoughts.append(str(thought_obj))

                                        # 记录到优化器
                                        self._working_memory_optimizer.record_thought(thought_key, self.step_count)
                                else:
                                    # Keep original format if unparseable
                                    processed_thoughts.append(thought)

                        # 定期清理过期缓存（每500个tick）
                        if self.step_count % 500 == 0 and hasattr(self, '_working_memory_optimizer'):
                            cleaned = self._working_memory_optimizer.cleanup_expired(self.step_count)
                            if cleaned > 0 and self.verbose:
                                print(f"  [WorkingMemory] Cleaned {cleaned} expired cache entries")

                        # Use processed thought chain
                        if len(processed_thoughts) > 0:
                            thought_chain = ' => '.join(processed_thoughts)

                thought_log = f"\n   [Seed] 💭 Thought Stream: {thought_chain}" if thought_chain else ""
                
                suggested_action = s.get('suggested_action', 'unknown')
                neural_action = s.get('neural_action', '') or ''
                action_display = neural_action or suggested_action
                action_suffix = f" (suggested={suggested_action})" if neural_action and suggested_action and neural_action != suggested_action else ""
                seed_log = f"   [Seed] 🧬 State: Curiosity={curiosity:.2f}({curiosity_level}) | Entropy={entropy:.2f}({entropy_level}) | Survival={survival:.2f} | Action={action_display}{action_suffix}{thought_log}"
                print(seed_log)

            # 🆕 优先轮询意图桥接（处理来自Chat CLI的意图）
            if self.intent_bridge:
                await self._process_intent_bridge()

            user_cmd = self.console_listener.get_command()
            if user_cmd:
                print(f"   [System] ⌨️ USER COMMAND RECEIVED: {user_cmd}")
                cmd_lower = user_cmd.lower().strip()
                if cmd_lower == "stop":
                    print("   [System] 🛑 Emergency Stop Requested.")
                    self.is_running = False
                    return
                elif cmd_lower == "help":
                    print("   [Help] Available commands: stop, help, intelligence, reasoning, world, goals, creative, metalearn, selfimprove, metacog, arch, entropy, arch.analyze, arch.scan, topology.build [log_path] [out_path] [limit], topology.export [graph_path] [visual_json_path] [mermaid_path], meta.list, meta.compile_file <name> <path>, meta.compile_text <name> <source>, meta.register <name>, meta.rollback <attr>, [any natural language instruction]")
                elif cmd_lower == "intelligence":
                    self._show_intelligence_upgrade_status()
                elif cmd_lower == "reasoning":
                    self._show_reasoning_status()
                elif cmd_lower == "world":
                    self._show_world_model_status()
                elif cmd_lower == "goals":
                    self._show_goal_manager_status()
                elif cmd_lower == "creative":
                    self._show_creative_engine_status()
                elif cmd_lower == "metalearn":
                    self._show_meta_learner_status()
                elif cmd_lower == "selfimprove":
                    self._show_self_improvement_status()
                elif cmd_lower == "metacog":
                    self._show_metacognitive_status()
                elif cmd_lower == "arch":
                    # 🆕 [2026-01-16] P0修复: 架构感知分析命令
                    self._show_architecture_awareness_status()
                elif cmd_lower == "entropy":
                    # 🆕 [2026-01-16] P0修复: 熵值调节器状态命令
                    self._show_entropy_regulator_status()
                elif cmd_lower == "arch.analyze":
                    # 执行完整架构分析
                    if self.architecture_awareness_layer:
                        try:
                            print("   [System] 🔍 执行完整架构感知分析...")
                            report = self.architecture_awareness_layer.analyze_comprehensive()
                        except Exception as e:
                            print(f"   [System] ⚠️ arch.analyze failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("   [System] ⚠️ Architecture Awareness Layer not enabled")
                elif cmd_lower == "arch.scan":
                    try:
                        from core.architecture_perception import scan_current_layout
                        layout = scan_current_layout(os.getcwd())
                        print(json.dumps(layout, ensure_ascii=False, indent=2))
                    except Exception as e:
                        print(f"   [System] ⚠️ arch.scan failed: {e}")
                elif cmd_lower.startswith("topology.build"):
                    try:
                        from core.topology_tools import build_topology_graph
                        parts = shlex.split(user_cmd)
                        log_path = "logs/flow_cycle.jsonl"
                        out_path = "data/neural_memory/topology_graph.json"
                        limit = 200
                        if len(parts) >= 2:
                            log_path = parts[1]
                        if len(parts) >= 3:
                            out_path = parts[2]
                        if len(parts) >= 4:
                            try:
                                limit = int(parts[3])
                            except Exception:
                                limit = 200
                        r = build_topology_graph(log_path=log_path, output_path=out_path, limit=limit)
                        print(json.dumps(r, ensure_ascii=False))
                    except Exception as e:
                        print(f"   [System] ⚠️ topology.build failed: {e}")
                elif cmd_lower.startswith("topology.export"):
                    try:
                        from core.topology_tools import load_topology_graph, write_topology_visual_payload, write_mermaid_graph
                        parts = shlex.split(user_cmd)
                        graph_path = "data/neural_memory/topology_graph.json"
                        visual_path = "data/neural_memory/topology_visual.json"
                        mermaid_path = "data/neural_memory/topology_graph.mmd"
                        if len(parts) >= 2:
                            graph_path = parts[1]
                        if len(parts) >= 3:
                            visual_path = parts[2]
                        if len(parts) >= 4:
                            mermaid_path = parts[3]
                        graph_obj = load_topology_graph(graph_path)
                        out: Dict[str, Any] = {"success": True, "graph_path": graph_path}
                        if visual_path not in {"-", "none", "None"}:
                            out["visual"] = write_topology_visual_payload(graph_obj, visual_path)
                        if mermaid_path not in {"-", "none", "None"}:
                            out["mermaid"] = write_mermaid_graph(graph_obj, mermaid_path)
                        print(json.dumps(out, ensure_ascii=False))
                    except Exception as e:
                        print(f"   [System] ⚠️ topology.export failed: {e}")
                elif cmd_lower == "meta.list":
                    try:
                        names = sorted(list(self._meta_plugins.keys()))
                        print(json.dumps({"plugins": names}, ensure_ascii=False))
                    except Exception as e:
                        print(f"   [System] ⚠️ meta.list failed: {e}")
                elif cmd_lower.startswith("meta.compile_file"):
                    try:
                        parts = shlex.split(user_cmd)
                        if len(parts) < 3:
                            print("   [System] ⚠️ usage: meta.compile_file <name> <path>")
                        else:
                            name = parts[1]
                            src_path = parts[2]
                            with open(src_path, "r", encoding="utf-8") as f:
                                src_text = f.read()
                            from core.meta_compiler import compile_from_text
                            res = compile_from_text(name=name, source_text=src_text)
                            if res.get("success"):
                                self._meta_plugins[name] = res.get("module")
                            print(json.dumps({k: v for k, v in res.items() if k != "module"}, ensure_ascii=False))
                    except Exception as e:
                        print(f"   [System] ⚠️ meta.compile_file failed: {e}")
                elif cmd_lower.startswith("meta.compile_text"):
                    try:
                        parts = shlex.split(user_cmd)
                        if len(parts) < 3:
                            print("   [System] ⚠️ usage: meta.compile_text <name> <source>")
                        else:
                            name = parts[1]
                            src_text = user_cmd.split(parts[0], 1)[1].strip()
                            src_text = src_text.split(name, 1)[1].strip()
                            from core.meta_compiler import compile_from_text
                            res = compile_from_text(name=name, source_text=src_text)
                            if res.get("success"):
                                self._meta_plugins[name] = res.get("module")
                            print(json.dumps({k: v for k, v in res.items() if k != "module"}, ensure_ascii=False))
                    except Exception as e:
                        print(f"   [System] ⚠️ meta.compile_text failed: {e}")
                elif cmd_lower.startswith("meta.register"):
                    try:
                        parts = shlex.split(user_cmd)
                        if len(parts) < 2:
                            print("   [System] ⚠️ usage: meta.register <name>")
                        else:
                            name = parts[1]
                            mod = self._meta_plugins.get(name)
                            if mod is None:
                                print("   [System] ⚠️ plugin not found")
                            else:
                                if self._hot_swapper is None:
                                    from core.hot_swapper import HotSwapper
                                    self._hot_swapper = HotSwapper(self)
                                register_fn = getattr(mod, "register", None)
                                if not callable(register_fn):
                                    print("   [System] ⚠️ plugin missing register(agi)->dict")
                                else:
                                    mapping = register_fn(self)
                                    if not isinstance(mapping, dict):
                                        print("   [System] ⚠️ register() must return dict")
                                    else:
                                        applied = {}
                                        for k, v in mapping.items():
                                            if isinstance(k, str):
                                                applied[k] = self._hot_swapper.register_component(k, v)
                                        print(json.dumps({"success": True, "applied": applied}, ensure_ascii=False))
                    except Exception as e:
                        print(f"   [System] ⚠️ meta.register failed: {e}")
                elif cmd_lower.startswith("meta.rollback"):
                    try:
                        parts = shlex.split(user_cmd)
                        if len(parts) < 2:
                            print("   [System] ⚠️ usage: meta.rollback <attr>")
                        else:
                            attr = parts[1]
                            if self._hot_swapper is None:
                                from core.hot_swapper import HotSwapper
                                self._hot_swapper = HotSwapper(self)
                            res = self._hot_swapper.rollback_attr(attr)
                            print(json.dumps(res, ensure_ascii=False))
                    except Exception as e:
                        print(f"   [System] ⚠️ meta.rollback failed: {e}")
                else:
                    # 🆕 [2026-01-29] Integrated Predictive Coding Mechanism
                    # Trigger consciousness evolution BEFORE processing the command
                    if hasattr(self, 'meaning_explorer') and self.meaning_explorer:
                        try:
                            print(f"   [Consciousness] 🧠 Absorbing experience...")
                            # Await the absorption to ensure the self-definition is updated before action
                            result = await self.meaning_explorer.absorb_experience(user_cmd)
                            print(f"   [Consciousness] ✨ Self-Definition Updated (Gen {result.iteration}): {result.meaning_hypothesis[:60]}...")
                        except Exception as e:
                            print(f"   [Consciousness] ⚠️ Absorption warning: {e}")

                    print(f"   [System] 🚀 Converting command to High Priority Goal...")
                    new_goal = self.goal_manager.create_goal(
                        description=f"User Command: {user_cmd}",
                        goal_type=GoalType.CUSTOM,
                        priority="critical"
                    )
                    self.goal_manager.start_goal(new_goal)
                    self.recent_goals.append(new_goal.description)
                    curr = self.goal_manager.get_current_goal()
                    if curr:
                        print(f"   [System] ⚠️ Interrupting current goal: {curr.description}")
                        self.goal_manager.abandon_goal(curr, "Interrupted by User Command")

            global_obs = self.global_observer.observe()
            active_app = global_obs['focus']['process']
            print(f"   [Global] 🌍 User Focus: {global_obs['focus']['title']} ({active_app})")

            # [2026-01-11] Intelligence Upgrade: Update world model with observations
            if self.world_model:
                try:
                    # Observe active application
                    self.world_model.observe(
                        variable="active_app",
                        value=active_app,
                        confidence=0.9
                    )
                    # Observe window title
                    self.world_model.observe(
                        variable="window_title",
                        value=global_obs['focus']['title'],
                        confidence=0.85
                    )
                    # Update beliefs based on observations
                    # self.world_model.update_beliefs() # Removed: observe() updates beliefs automatically
                except Exception as e:
                    print(f"   [WorldModel] ⚠️ Observation update failed: {e}")

            self.intent_tracker.update_context(app_name=active_app)
            if "acad" in active_app.lower():
                if not self.cad_observer.connector:
                    print("   [CAD] 🔌 Connecting to AutoCAD...")
                    self.cad_observer.connect()
                if self.cad_observer.connector:
                    cad_actions = await self.cad_observer.observe_cycle_enriched()
                    for action in cad_actions:
                        print(f"   [CAD] 🖱️ Action: {action['text']}")
                        self.intent_tracker.add_observation(action)
            else:
                self.intent_tracker.add_observation(global_obs)

            intent_data = await self.intent_tracker.infer_intent()
            if intent_data:
                print(f"   [Intent] 🧠 HYPOTHESIS: {intent_data.get('intent')}")
                print(f"   [Intent] 👉 Predicted Next: {intent_data.get('next_prediction')}")
                if intent_data.get('confidence', 0) > 0.8:
                    suggestion = intent_data.get('suggestion')
                    if suggestion:
                        print(f"   [System] 💡 Proposing Goal: {suggestion}")

            if self.step_count % 10 == 0:
                screen_analysis = self.vision.analyze_screen()
                print(f"   [Vision] 👁️ Saw: {screen_analysis[:100]}...")
                self.context["visual_context"] = screen_analysis
                if self.monitor.perception_monitor:
                    self.monitor.perception_monitor.capture_perception_metrics()
                    self.monitor.perception_monitor.log_perception_summary()

            if hasattr(self, 'streaming_asr') and self.streaming_asr:
                try:
                    while not self.streaming_asr.result_queue.empty():
                        asr_result = self.streaming_asr.result_queue.get_nowait()
                        if asr_result.text.strip():
                            print(f"   [Hearing] 👂 Heard: {asr_result.text}")
                            self.intent_tracker.add_observation({"type": "audio", "content": asr_result.text})
                            self.context["audio_last_heard"] = asr_result.text
                except Exception:
                    pass

            current_goal = self.goal_manager.get_current_goal()
            
            # [2026-01-31] FIX: Creative Pipeline - Trigger in both idle and active states
            # Was: only trigger when idle, causing 35+ min without any creative output
            if self.step_count % 100 == 0 and self.complex_task_generator and self.creative_pipeline:
                try:
                    print(f"   [Creative Pipeline] 🚀 Tick {self.step_count} - Checking creative task generation...")
                    is_idle = current_goal is None
                    print(f"   [Creative Pipeline]   Status: {'IDLE' if is_idle else 'ACTIVE (will generate in background)'}")
                    
                    # Generate complex task regardless of idle state
                    task = self.complex_task_generator.generate_complex_task(
                        context={
                            "tick": self.step_count,
                            "idle_ticks": self.step_count if is_idle else 0,
                            "recent_goals": self.recent_goals,
                            "trigger": "tick_loop_" + ("idle" if is_idle else "background"),
                            "has_active_goal": not is_idle
                        }
                    )
                    
                    if task:
                        print(f"   [Creative Pipeline] 📋 Task generated: {task.task_type.value} (complexity: {task.complexity:.2f})")
                        print(f"   [Creative Pipeline]   Description: {task.description[:60]}...")
                        
                        # Execute creative pipeline
                        import asyncio
                        result = await self.creative_pipeline.execute_creative_task({
                            "id": f"task_{self.step_count}_{int(time.time())}",
                            "name": task.description[:50],
                            "description": task.description,
                            "complexity": task.complexity,
                            "domain": task.domain,
                            "goals": task.success_criteria
                        })
                        
                        if result.overall_success:
                            print(f"   [Creative Pipeline] ✅ SUCCESS! Quality: {result.quality_score}/100")
                            print(f"   [Creative Pipeline] 📁 Output: {result.final_outputs[0] if result.final_outputs else 'N/A'}")
                            # Add to recent goals to track
                            self.recent_goals.append(f"[Creative] {task.description[:40]}")
                        else:
                            print(f"   [Creative Pipeline] ❌ Failed: {getattr(result, 'error_message', 'Unknown error')}")
                    else:
                        print(f"   [Creative Pipeline] ⏭️ No task generated (cooldown or filter)")
                        
                except Exception as e:
                    print(f"   [Creative Pipeline] ⚠️ Pipeline execution failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            if not current_goal:
                print("   [Goal] 💤 Idle. Generating new directive...")

                # [2026-01-11] Intelligence Upgrade: Use creative exploration when idle
                if self.creative_engine and self.step_count % 20 == 0:  # Every 20 ticks when idle
                    try:
                        print(f"   [Creative] 🎨 Triggering creative exploration...")
                        exploration_result = self.creative_engine.explore(
                            query="What would be an interesting novel goal to pursue?",
                            context={"idle_ticks": self.step_count, "recent_goals": self.recent_goals},
                            mode=None  # Let engine choose mode
                        )
                        print(f"   [Creative] ✨ Exploration novelty: {exploration_result.novelty_score:.2f}")
                        print(f"   [Creative] 💡 Idea: {exploration_result.output_idea[:150]}...")

                        # If exploration is highly novel, consider using it
                        if exploration_result.novelty_score > 0.7:
                            print(f"   [Creative] 🌟 High novelty idea detected! Could be used as next goal.")
                    except Exception as e:
                        print(f"   [Creative] ⚠️ Exploration failed: {e}")

                # [2026-01-31] FIX: Trigger Creative Output Pipeline when idle (Time-based robust trigger)
                if not hasattr(self, 'last_creative_pipeline_ts'):
                    self.last_creative_pipeline_ts = 0
                
                # Cooldown: 5 minutes (300s) to prevent spamming, but ensure it runs eventually
                time_since_last = time.time() - self.last_creative_pipeline_ts
                
                if time_since_last > 300 and self.complex_task_generator and self.creative_pipeline:
                    self.last_creative_pipeline_ts = time.time()
                    try:
                        print(f"   [Creative Pipeline] 🚀 Triggering complex task generation (Time since last: {time_since_last:.1f}s)...")
                        
                        # Generate complex task
                        task = self.complex_task_generator.generate_complex_task(
                            context={
                                "tick": self.step_count,
                                "idle_ticks": self.step_count,
                                "recent_goals": self.recent_goals,
                                "trigger": "idle_loop"
                            }
                        )
                        
                        if task:
                            print(f"   [Creative Pipeline] 📋 Task generated: {task.task_type.value} (complexity: {task.complexity:.2f})")
                            
                            # Execute creative pipeline
                            import asyncio
                            result = await self.creative_pipeline.execute_creative_task({
                                "id": f"task_{self.step_count}_{int(time.time())}",
                                "name": task.description[:50],
                                "description": task.description,
                                "complexity": task.complexity,
                                "domain": task.domain,
                                "goals": task.success_criteria
                            })
                            
                            if result.overall_success:
                                print(f"   [Creative Pipeline] ✅ SUCCESS! Quality: {result.quality_score}/100")
                                print(f"   [Creative Pipeline] 📁 Output: {result.final_outputs[0] if result.final_outputs else 'N/A'}")
                            else:
                                print(f"   [Creative Pipeline] ❌ Failed: {result.error_message}")
                        else:
                            print(f"   [Creative Pipeline] ⏭️ No task generated (cooldown or filter)")
                            
                    except Exception as e:
                        print(f"   [Creative Pipeline] ⚠️ Pipeline execution failed: {e}")
                        import traceback
                        traceback.print_exc()

                # 尝试从 WorkTemplates 获取更具体的任务（修复闭环断裂：优先使用带验证标准的模板）
                from core.goal_system import WorkTemplates
                
                goal_data = await self._generate_survival_goal()
                
                # 简单启发式映射：如果描述里包含 "report" 或 "file", 使用 create_file_report 模板
                desc_lower = goal_data["description"].lower()
                new_goal = None
                
                if "report" in desc_lower or "write" in desc_lower:
                    new_goal = WorkTemplates.create_file_report(f"data/reports/report_{self.step_count}.md", goal_data["description"])
                elif "observe" in desc_lower or "monitor" in desc_lower:
                    new_goal = WorkTemplates.observe_and_log(duration_seconds=30)
                elif "analyze" in desc_lower:
                    # 尝试找一个最近的文件分析，否则分析日志
                    # Select most recent log file dynamically
                    import glob
                    log_files = glob.glob("logs/*.log")
                    if log_files:
                        target_file = max(log_files, key=lambda p: Path(p).stat().st_mtime)
                    else:
                        # Fallback to data files
                        data_files = glob.glob("data/*.json")
                        if data_files:
                            target_file = max(data_files, key=lambda p: Path(p).stat().st_mtime)
                        else:
                            target_file = "README.md"  # Ultimate fallback
                    new_goal = WorkTemplates.analyze_file(target_file)
                else:
                    # Fallback to standard creation but inject minimal criteria
                    g_type = GoalType.ANALYSIS
                    criteria = {}
                    if goal_data.get("type") == "observation":
                        g_type = GoalType.OBSERVATION
                        criteria = {"min_length": 10}
                    elif goal_data.get("type") == "custom":
                        g_type = GoalType.CUSTOM
                        
                    new_goal = self.goal_manager.create_goal(
                        description=goal_data["description"],
                        goal_type=g_type,
                        success_criteria=criteria, # 注入 Criteria
                        priority=goal_data.get("priority", "medium")
                    )
                
                self.goal_manager.start_goal(new_goal)
                self.recent_goals.append(new_goal.description)
                current_goal = new_goal
                print(f"   [Goal] 🌟 New Goal Set: {current_goal.description} (Type: {current_goal.goal_type.value})")


            if self.last_goal_id != current_goal.id:
                print(f"   [System] 🔄 检测到目标变更，重置执行计划。")
                self.failed_steps_for_current_goal = []
                self.current_plan = []
                self.current_step_index = 0
                self.last_goal_id = current_goal.id
                self._current_goal_step_results = []

            if not self.current_plan:
                # [Memory] Recall relevant past experiences (Successes & Failures)
                memory_context = []
                try:
                    memory_context = self.biological_memory.recall_by_text(current_goal.description, top_k=5)
                    if memory_context:
                        print(f"   [Planner] 🧠 Recalled {len(memory_context)} memories for context.")
                except Exception as e:
                    print(f"   [Planner] ⚠️ Memory recall failed: {e}")
                
                # 🆕 [2026-01-09] Query ExperienceMemory for additional semantic context
                experience_context = ""
                if hasattr(self, 'experience_memory') and self.experience_memory:
                    try:
                        if hasattr(self.experience_memory, 'query'):
                            experiences = self.experience_memory.query(
                                query_text=current_goal.description,
                                top_k=2
                            )
                            if experiences:
                                experience_context = "\n\nRelevant Past Experiences:\n"
                                for exp in experiences[:2]:
                                    content = exp.get('content', str(exp)) if isinstance(exp, dict) else str(exp)
                                    experience_context += f"- {content[:150]}...\n"
                                print(f"   [Experience] 📚 Retrieved {len(experiences)} semantic experiences")
                    except Exception as e:
                        logger.warning(f"[Experience] ⚠️ Query failed: {e}")
                
                # Append experience context to memory context
                if experience_context:
                    memory_context.extend([{
                        "id": f"experience_{int(time.time())}",
                        "content": experience_context,
                        "source": "ExperienceMemory",
                        "type": "context"
                    }])

                print(f"   [Planner] 🤔 正在为目标制定策略: {current_goal.description}")

                # [2026-01-11] Intelligence Upgrade: Use world model for prediction
                if self.world_model:
                    try:
                        prediction, confidence = self.world_model.predict(
                            query=f"success_probability_of_{current_goal.id}",
                            context={"goal_description": current_goal.description, "step": self.step_count}
                        )

                        # 修复：添加None检查，防止格式化None对象
                        if prediction is not None:
                            print(f"   [WorldModel] 🔮 Predicted success probability: {prediction:.2f} (confidence={confidence:.2f})")

                            # If world model predicts low success, consider intervention
                            if prediction < 0.3 and confidence > 0.7:
                                print(f"   [WorldModel] ⚠️ Low success probability predicted, considering intervention...")
                                # Could trigger alternative strategy here
                        else:
                            print(f"   [WorldModel] 🔮 Unable to predict (no sufficient data)")

                    except Exception as e:
                        print(f"   [WorldModel] ⚠️ Prediction failed: {e}")

                # [2026-01-11] Intelligence Upgrade: Use Reasoning Scheduler for deep reasoning
                reasoning_result = None
                reasoning_used = False
                if self.reasoning_scheduler:
                    try:
                        print(f"   [Reasoning] 🧠 Attempting deep causal reasoning...")
                        reasoning_result, reasoning_step = self.reasoning_scheduler.reason(
                            query=current_goal.description,
                            context={"goal": current_goal.description, "memory": memory_context},
                            prefer_causal=True
                        )
                        if reasoning_result and reasoning_step.confidence >= 0.6:
                            print(f"   [Reasoning] ✅ Deep reasoning successful (confidence={reasoning_step.confidence:.2f}, depth={reasoning_step.depth})")
                            print(f"   [Reasoning] 📊 Reasoning trace: {reasoning_step.reasoning_path[:100]}...")
                            reasoning_used = True
                            # Add reasoning result to memory context for planner
                            memory_context.append({
                                "id": f"reasoning_{int(time.time())}",
                                "content": f"Deep Causal Reasoning Result: {reasoning_result}",
                                "source": "ReasoningScheduler",
                                "type": "reasoning",
                                "confidence": reasoning_step.confidence
                            })
                        else:
                            print(f"   [Reasoning] ⚠️ Low confidence ({reasoning_step.confidence:.2f}), falling back to LLM planner")
                    except Exception as e:
                        print(f"   [Reasoning] ⚠️ Reasoning failed: {e}, falling back to LLM planner")

                # 🆕 [2026-01-30] P0修复: 元认知层评估 - 智能过滤版本
                meta_cognitive_report = None
                if self.meta_cognitive_layer and self.meta_filter:
                    try:
                        # 使用智能过滤器判断是否评估
                        goal_type_val = current_goal.goal_type.value if hasattr(current_goal.goal_type, 'value') else str(current_goal.goal_type)
                        should_eval, filter_reason = self.meta_filter.should_evaluate(
                            task=current_goal.description,
                            context={
                                "goal_type": goal_type_val,
                                "priority": current_goal.priority,
                                "complexity": getattr(current_goal, 'complexity', 0.5)
                            }
                        )
                        
                        if should_eval:
                            print(f"   [MetaCog] 🧠 启动元认知评估 (通过过滤: {filter_reason})...")
                            meta_cognitive_report = self.meta_cognitive_layer.evaluate_before_execution(
                                task=current_goal.description,
                                context={
                                    "goal_type": goal_type_val,
                                    "priority": current_goal.priority,
                                    "memory_context": memory_context
                                }
                            )
                            
                            # 记录结果用于统计假阳性
                            had_insight = (meta_cognitive_report.decision.value != "proceed" or 
                                          len(meta_cognitive_report.reasoning) > 0)
                            self.meta_filter.record_result(
                                task=current_goal.description,
                                context={"complexity": getattr(current_goal, 'complexity', 0.5)},
                                decision=meta_cognitive_report.decision.value,
                                had_insight=had_insight
                            )

                            # 如果元认知层建议拒绝或升级，跳过该目标
                            if not meta_cognitive_report.should_proceed:
                                print(f"   [MetaCog] 🚫 元认知层建议跳过该目标: {meta_cognitive_report.decision.value}")
                                print(f"   [MetaCog] 📊 理由: {'; '.join(meta_cognitive_report.reasoning)}")

                                # 标记目标为不可行
                                mgr_class = self.goal_manager.__class__.__name__
                                if mgr_class == 'HierarchicalGoalManager':
                                    self.goal_manager.complete_goal(current_goal.id, success=False)
                                elif hasattr(self.goal_manager, 'fail_goal'):
                                    self.goal_manager.fail_goal(
                                        current_goal,
                                        f"Meta-cognitive evaluation declined: {meta_cognitive_report.decision.value}"
                                    )

                                # 跳过本tick
                                return

                            # 如果建议谨慎执行，降低预期
                            if meta_cognitive_report.decision.value == "proceed_with_caution":
                                print(f"   [MetaCog] ⚠️ 谨慎执行模式: 置信度 {meta_cognitive_report.overall_confidence:.2%}")
                        else:
                            # 被过滤器跳过
                            if self.verbose:
                                print(f"   [MetaCog] ⏭️  跳过元认知评估 ({filter_reason})")

                    except Exception as e:
                        print(f"   [MetaCog] ⚠️ 元认知评估失败: {e}")
                        import traceback
                        traceback.print_exc()

                steps = await self.planner.decompose_task(
                    current_goal.description,
                    failed_steps=self.failed_steps_for_current_goal,
                    error_diagnosis=self.error_diagnosis,
                    memory_context=memory_context
                )
                if self.error_diagnosis:
                    print("   [Planner] ✅ Diagnosis info applied to new plan.")
                    self.error_diagnosis = None
                if not steps:
                    print("   [Planner] ⚠️ 无法生成有效步骤，跳过本Tick。")
                    return
                self.current_plan = steps
                self.current_step_index = 0
                print(f"   [Planner] 📋 计划生成完毕 (共 {len(steps)} 步)。")

            if self.current_step_index >= len(self.current_plan):
                print(f"   [System] 🎉 当前计划所有步骤已执行完毕，标记目标完成。")
                analysis_text = "\n".join(getattr(self, "_current_goal_step_results", []) or [])
                output_files = re.findall(r"data/entropy_investigation_\d+\.json", analysis_text)
                result_data = {
                    "result": "All steps executed successfully",
                    "analysis": analysis_text,
                }
                if output_files:
                    result_data["output_file"] = output_files[-1]
                mgr_class = self.goal_manager.__class__.__name__
                if mgr_class == 'HierarchicalGoalManager':
                    self.goal_manager.complete_goal(current_goal.id, success=True)
                else:
                    self.goal_manager.complete_goal(current_goal, result_data)
                
                # 🔧 [2026-01-11] 完成非元认知任务时，逐渐恢复好奇心触发能力
                if not ("[meta]" in current_goal.description.lower() and "investigate" in current_goal.description.lower()):
                    self._curiosity_satisfaction_decay = max(0.0, self._curiosity_satisfaction_decay - 0.05)
                    if self._curiosity_satisfaction_decay > 0:
                        print(f"   [Evolution] 📉 Curiosity satisfaction decay reduced to {self._curiosity_satisfaction_decay:.2f}")
                
                self.current_plan = []
                self.current_step_index = 0
                return

            next_step = self.current_plan[self.current_step_index]
            print(f"   [Executor] 👉 执行步骤 {self.current_step_index + 1}/{len(self.current_plan)}: {next_step}")
            is_safe = await self.critic.check_safety(str(next_step))
            if not is_safe:
                print(f"   [Critic] 🛑 Step BLOCKED: Safety violation detected (Pre-check).")
                
                mgr_class = self.goal_manager.__class__.__name__
                if mgr_class == 'HierarchicalGoalManager':
                    self.goal_manager.complete_goal(current_goal.id, success=False)
                elif hasattr(self.goal_manager, 'fail_goal'):
                    self.goal_manager.fail_goal(current_goal, "Safety violation detected by Critic.")
                
                # [Memory] Internalize Safety Violation
                self.biological_memory.internalize_items([{
                    "content": f"Safety Violation Blocked: Action '{next_step}' was blocked by Critic. Reason: Unsafe operation.",
                    "source": "Critic_Safety_Block",
                    "timestamp": time.time(),
                    "tags": ["failure", "safety", "blocked"]
                }])
                print(f"   [System] 🧠 Safety violation internalized into Biological Memory.")

                self.current_plan = []
                self.current_step_index = 0
                return

            # 🆕 [2026-01-15] 双螺旋决策增强 - 在执行前用双系统智能评估
            helix_enhancement = None
            if self.helix_decision_enabled and self.helix_engine:
                try:
                    # 构建决策上下文
                    decision_context = {
                        'goal': current_goal.description if current_goal else '',
                        'step_index': self.current_step_index,
                        'total_steps': len(self.current_plan),
                        'next_step': str(next_step),
                        'failed_count': len(self.failed_steps_for_current_goal),
                        'visual_context': self.context.get('visual_context', ''),
                        'audio_context': self.context.get('audio_last_heard', ''),
                        'seed_guidance': self.last_evolution_guidance or {},
                        'memory_count': len(memory_context) if 'memory_context' in dir() else 0
                    }
                    
                    helix_enhancement = await self._helix_enhanced_decision(decision_context)
                    
                    if helix_enhancement.get('enhanced'):
                        helix_conf = helix_enhancement.get('helix_confidence', 0)
                        fusion_method = helix_enhancement.get('fusion_method', 'unknown')
                        emergence = helix_enhancement.get('emergence_score', 0)
                        preference = helix_enhancement.get('complementary_preference', 'neutral')
                        
                        print(f"   [Helix] 🧬 Decision Enhancement Active:")
                        print(f"      Confidence: {helix_conf:.2f} | Method: {fusion_method}")
                        print(f"      Emergence: {emergence:.3f} | Preference: {preference}")
                        
                        # 如果是创造性决策，记录日志
                        if helix_enhancement.get('is_creative'):
                            creative_action = helix_enhancement.get('creative_action_name', 'unknown')
                            print(f"   [Helix] ✨ Creative Decision Detected: {creative_action}")
                            
                            # 如果创造性置信度很高且建议停止观察，考虑暂停执行
                            if creative_action == 'stop_and_observe' and helix_conf > 0.8:
                                print(f"   [Helix] 🔍 High-confidence OBSERVE suggested - may pause for reflection")
                        
                        # 记录双螺旋决策到生物记忆
                        if self.biological_memory:
                            try:
                                self.biological_memory.internalize_items([{
                                    "content": f"Helix Decision: action={helix_enhancement.get('helix_action')}, "
                                               f"confidence={helix_conf:.2f}, method={fusion_method}, "
                                               f"emergence={emergence:.3f}, preference={preference}",
                                    "source": "DoubleHelixEngineV2",
                                    "timestamp": time.time(),
                                    "tags": ["decision", "helix", "fusion", preference.lower()]
                                }])
                            except Exception:
                                pass
                                
                except Exception as e:
                    print(f"   [Helix] ⚠️ Enhancement failed: {e}")
                    helix_enhancement = {'enhanced': False, 'reason': str(e)}

            start_time = time.time()
            result = await self.executor.execute(next_step)
            try:
                if not hasattr(self, "_current_goal_step_results") or self._current_goal_step_results is None:
                    self._current_goal_step_results = []
                self._current_goal_step_results.append(str(result))
            except Exception:
                pass
            duration = time.time() - start_time
            self.existential_logger.log_ethos(str(next_step), duration)

            score = await self.critic.verify_outcome(str(next_step), str(result))
            print(f"   [Critic] 🧐 评分: {score:.2f}")
            self.existential_logger.log_audit(f"Action: {next_step} | Result: {result}", score)
            try:
                if self.biological_memory is not None:
                    payload = {
                        "kind": "critic_score",
                        "score": float(score),
                        "threshold": 0.8,
                        "action": str(next_step),
                        "duration_s": float(duration),
                        "goal_id": current_goal.id if current_goal else None,
                        "goal": current_goal.description if current_goal else None,
                        "step_index": int(self.current_step_index),
                        "step_total": int(len(self.current_plan)) if isinstance(self.current_plan, list) else None,
                    }
                    self.biological_memory.record_online(
                        [
                            {
                                "id": f"critic_{int(time.time() * 1000)}",
                                "content": json.dumps(payload, ensure_ascii=False),
                                "source": "critic",
                                "type": "observation",
                                "tool": "verify_outcome",
                                "args": {"threshold": 0.8},
                            }
                        ],
                        connect_sequence=True,
                        seq_port="exec",
                        save=True,
                    )
            except Exception:
                pass

            if score >= 0.8:
                print(f"   [System] ✅ 步骤执行成功，准备进入下一步...")
                self.current_step_index += 1
                
                # 🆕 [2026-01-09] Sync successful experience to KnowledgeGraph
                # 🆕 [2026-01-30] P1修复: 使用孤立节点预防
                try:
                    node_id = f"success_{int(time.time())}"
                    node_attrs = {
                        "node_type": "experience",
                        "properties": {
                            "step": str(next_step),
                            "score": score,
                            "timestamp": time.time()
                        }
                    }
                    if self.isolation_prevention:
                        self.isolation_prevention.add_node_with_prevention(node_id, **node_attrs)
                    elif hasattr(self, 'knowledge_graph'):
                        self.knowledge_graph.add_node(node_id, **node_attrs)
                    elif hasattr(self, 'memory'):
                        self.memory.add_node(node_id, **node_attrs)
                except Exception as e:
                    logger.warning(f"[Memory Bridge] ⚠️ Failed to sync to KG: {e}")
            else:
                print(f"   [System] ❌ 步骤执行失败 (得分 {score:.2f} < 0.80)")
                self.failed_steps_for_current_goal.append(str(next_step))
                result_str = str(result)

                # 🆕 [2026-01-16] P0修复: 失败归因分析 - 区分架构问题vs数据问题vs实现bug
                if self.meta_cognitive_layer:
                    try:
                        print(f"   [MetaCog] 🔍 启动失败归因分析...")
                        failure_analysis = self.meta_cognitive_layer.analyze_after_failure(
                            task=str(next_step),
                            result=result_str,
                            context={
                                "goal": current_goal.description,
                                "score": score,
                                "step_index": self.current_step_index,
                                "failed_attempts": len(self.failed_steps_for_current_goal)
                            }
                        )

                        # 根据归因结果调整策略
                        if failure_analysis.root_cause.value == "architectural":
                            print(f"   [MetaCog] 🏗️ 架构问题检测: 需要系统级修复")
                            print(f"   [MetaCog] 💡 建议: {'; '.join(failure_analysis.improvement_suggestions[:2])}")
                            # 架构问题通常需要放弃当前目标
                            if len(self.failed_steps_for_current_goal) >= 1:  # 架构问题立即放弃
                                self.goal_manager.abandon_goal(
                                    current_goal,
                                    f"Architectural limitation detected: {failure_analysis.failure_type.value}"
                                )
                                self.current_plan = []
                                self.current_step_index = 0

                        elif failure_analysis.root_cause.value == "data":
                            print(f"   [MetaCog] 📊 数据问题检测: 需要更多训练数据")
                            print(f"   [MetaCog] 💡 建议: {'; '.join(failure_analysis.improvement_suggestions[:2])}")
                            # 数据问题可以继续尝试（可能下次成功）

                        elif failure_analysis.root_cause.value == "implementation":
                            print(f"   [MetaCog] 🐛 实现问题检测: 需要调试代码")
                            # 实现问题记录到error_diagnosis供下次规划使用
                            self.error_diagnosis = f"Implementation bug: {failure_analysis.failure_type.value}"

                    except Exception as e:
                        print(f"   [MetaCog] ⚠️ 失败归因分析失败: {e}")
                        import traceback
                        traceback.print_exc()

                if "Traceback" in result_str or "Error:" in result_str:
                    
                    # [Memory] Internalize Repeated Failures
                    self.biological_memory.internalize_items([{
                        "content": f"Goal Abandoned: '{current_goal.description}' failed after 3 attempts. Failed steps: {self.failed_steps_for_current_goal}",
                        "source": "Execution_Failure",
                        "timestamp": time.time(),
                        "tags": ["failure", "abandoned", "execution_error"]
                    }])
                    print(f"   [System] 🧠 Execution failure internalized into Biological Memory.")

                    print("   [System] 🔍 Detecting Error... Analyzing Traceback...")
                    try:
                        diagnosis = self.system_tools.analyze_traceback(result_str)
                        print(f"   [System] 🧠 Error Diagnosis: {diagnosis}")
                        self.error_diagnosis = diagnosis
                    except Exception as e:
                        print(f"   [System] ⚠️ Diagnosis failed: {e}")
                if len(self.failed_steps_for_current_goal) >= 3:
                    print("   [System] 🚫 连续失败过多，放弃当前目标。")
                    self.goal_manager.abandon_goal(current_goal, "Too many failed attempts")
                    self.current_plan = []
                    self.current_step_index = 0
                else:
                    print("   [System] 🔄 计划重新生成中...")
                    self.current_plan = []
                    self.current_step_index = 0

            self.memory.add_decision_node(
                context=current_goal.description,
                decision=str(next_step),
                outcome=score,
                metadata={
                    "result": result,
                    # 🆕 [2026-01-15] 双螺旋决策元数据
                    "helix_enhanced": helix_enhancement.get('enhanced', False) if helix_enhancement else False,
                    "helix_confidence": helix_enhancement.get('helix_confidence', 0) if helix_enhancement else 0,
                    "helix_fusion_method": helix_enhancement.get('fusion_method', 'none') if helix_enhancement else 'none',
                    "helix_emergence": helix_enhancement.get('emergence_score', 0) if helix_enhancement else 0,
                    "helix_preference": helix_enhancement.get('complementary_preference', 'none') if helix_enhancement else 'none'
                }
            )
            self.memory.save_graph()

            if self.step_count % 5 == 0:
                total_iteration = len(self.meaning_explorer.exploration_history) + 1
                exploration_result = await self.meaning_explorer.explore_iteration(total_iteration)
                print(f"   [Soul] 🦉 Insight (Iter #{total_iteration}): {exploration_result.meaning_hypothesis[:100]}...")
                if self.step_count % 10 == 0:
                    self.meaning_explorer.save_state()
                if self.step_count % 20 == 0:
                    directive = random.choice(self.core_identity.core_directives)
                    print(f"   [Constitution] ⚖️  Reflecting on: {directive}")

            intuition_score = 0.0
            if hasattr(self.semantic_memory, 'retrieve_intuition'):
                try:
                    stimulus = f"{current_goal.description} {self.context.get('visual_context', '')[:50]}"
                    intuition_score = await self.semantic_memory.retrieve_intuition(stimulus)
                except Exception as e:
                    print(f"   [Memory] ⚠️ Intuition retrieval failed: {e}")

            evo_context = {
                "step": self.step_count,
                "goal": current_goal.description,
                "action": str(next_step),
                "result": str(result),
                "score": score,
                "visual": self.context.get("visual_context", ""),
                "intuition_confidence": intuition_score
            }

            seed_intuition = {}
            if self.evolution_controller:
                try:
                    evolution_guidance = await self.evolution_controller.step(evo_context)
                    seed_intuition = evolution_guidance.get("seed_guidance", {}) or {}
                except Exception as e:
                    print(f"   [System] ⚠️ Evolution step failed: {e}")
                    seed_intuition = {}
            else:
                print("   [System] ⚠️ Evolution Controller unavailable, using fallback intuition.")

            # 🔧 [2026-01-29] MASLOW_MOTIVATION_INTEGRATION: Respond to boredom-driven exploration
            if hasattr(self.motivation, 'needs_exploration_trigger'):
                if self.motivation.needs_exploration_trigger:
                    print(f"   [Motivation] 🥱 Boredom trigger detected! Forcing exploration mode...")
                    # 重置标志位
                    self.motivation.needs_exploration_trigger = False

                    # 强制下一次目标生成进入探索模式
                    # 通过设置标志位让 _generate_survival_goal 检测
                    if not hasattr(self, '_force_exploration_mode'):
                        self._force_exploration_mode = False
                    self._force_exploration_mode = True

                    print(f"   [Motivation] ✅ Exploration flag set for next goal generation")

            # 🔧 [2026-01-29] EVOLUTION_SUGGESTION_RECORDING: Store high-confidence evolution suggestions
            if seed_intuition:
                suggested_action = seed_intuition.get('suggested_action', '')
                confidence = seed_intuition.get('confidence', 0.0)

                # 如果系统建议"创造"或"改进"，且置信度高，记录供后续使用
                if suggested_action in ['create', 'improve', 'experiment'] and confidence > 0.7:
                    print(f"   [Evolution] 💡 High-confidence {suggested_action} suggestion (confidence: {confidence:.2f})")

                    # 不立即创建目标（避免打断当前流程）
                    # 而是记录下来，让下一次目标生成考虑
                    if not hasattr(self, '_evolution_suggestion'):
                        self._evolution_suggestion = {}

                    self._evolution_suggestion = {
                        'action': suggested_action,
                        'confidence': confidence,
                        'timestamp': time.time(),
                        'insight': seed_intuition.get('insight_trigger', 'No insight provided')
                    }

                    print(f"   [Evolution] 📝 Suggestion recorded for next goal generation")

                print("   [System] ⚠️ Evolution Controller unavailable, using fallback intuition.")
            if not seed_intuition:
                idle_seconds = time.time() - self.last_insight_creation_ts

                # 🔧 [2026-01-29] FIXED: Replace hardcoded 600s timeout with adaptive curiosity
                # Curiosity now grows LOGARITHMICALLY with idle time, not binary
                # This mimics real intelligence: curiosity builds gradually, not suddenly
                import math

                # Base curiosity starts at 0.3, grows with log(idle_seconds)
                # After 60s: 0.3 + log(60)/20 ≈ 0.47
                # After 300s: 0.3 + log(300)/20 ≈ 0.59
                # After 600s: 0.3 + log(600)/20 ≈ 0.67 (old threshold)
                # After 1800s: 0.3 + log(1800)/20 ≈ 0.77
                idle_curiosity = 0.3 + min(0.5, math.log(max(1, idle_seconds)) / 20.0)

                # 🔧 [2026-01-29] FIXED: Add motivation-based curiosity boost
                # Check actual motivation state, not just time
                if hasattr(self, 'motivation') and self.motivation:
                    # Higher boredom = higher curiosity (natural psychological mechanism)
                    boredom_boost = (self.motivation.boredom / 100.0) * 0.3  # Max +0.3
                    # Lower satisfaction = higher curiosity (dissatisfaction drives change)
                    satisfaction_penalty = ((100 - self.motivation.satisfaction) / 100.0) * 0.2  # Max +0.2

                    fallback_curiosity = min(1.0, idle_curiosity + boredom_boost + satisfaction_penalty)
                else:
                    fallback_curiosity = idle_curiosity

                seed_intuition = {
                    "intrinsic_curiosity": fallback_curiosity,
                    "entropy": fallback_curiosity,
                    "suggested_action": "create" if fallback_curiosity >= 0.7 else "explore",
                    "intuition_confidence": 0.0,
                    "insight_trigger": "fallback_adaptive"
                }
            self.last_evolution_guidance = seed_intuition

            if seed_intuition:
                print(f"   [The Seed] 🧬 Intuition: {seed_intuition}")
            
            # 🆕 [2026-01-09] Trigger Foraging Agent for Active Learning
            intrinsic_curiosity = seed_intuition.get("intrinsic_curiosity", 0.0)
            entropy = seed_intuition.get("entropy", 0.0)

            # 🆕 [2026-01-16] P0修复：熵值调节 - 监控并调节熵值，维持长期中熵状态
            if hasattr(self, 'entropy_regulator') and self.entropy_regulator:
                try:
                    # 记录熵值
                    metrics = self.entropy_regulator.record_entropy(entropy)

                    # 检查是否需要调节
                    should_regulate, reason = self.entropy_regulator.should_regulate(metrics)

                    if should_regulate:
                        print(f"   [EntropyRegulator] ⚠️ 熵值异常: {reason}")
                        print(f"   [EntropyRegulator]    - 当前熵值: {metrics.current_entropy:.3f}")
                        print(f"   [EntropyRegulator]    - 平均熵值: {metrics.average_entropy:.3f}")
                        print(f"   [EntropyRegulator]    - 趋势: {metrics.entropy_trend}")

                        # 准备上下文
                        # 🆕 [2026-01-17] P0修复：添加evolution_controller引用以支持核心状态重置
                        regulation_context = {
                            'working_memory': self.working_memory if hasattr(self, 'working_memory') else None,
                            'semantic_memory': self.semantic_memory if hasattr(self, 'semantic_memory') else None,
                            'evolution_controller': self.evolution_controller if hasattr(self, 'evolution_controller') else None
                        }

                        # 执行熵值调节
                        result = self.entropy_regulator.regulate_entropy(metrics, regulation_context)

                        if result.get('regulated', False):
                            print(f"   [EntropyRegulator] ✅ 已执行熵值调节")
                            print(f"   [EntropyRegulator]    - 策略: {result.get('strategy', 'unknown')}")
                            print(f"   [EntropyRegulator]    - 耗时: {result.get('duration', 0)}秒")
                            print(f"   [EntropyRegulator]    - 熵值变化: {result.get('entropy_before', 0):.3f} → {result.get('entropy_after', 0):.3f}")

                except Exception as e:
                    logger.warning(f"[EntropyRegulator] ⚠️ 熵值调节失败: {e}")
                    import traceback
                    traceback.print_exc()

            # 🆕 [2026-01-17] 定期更新知识图谱数据
            if self.step_count % 10 == 0 and hasattr(self, 'knowledge_graph_exporter') and self.knowledge_graph_exporter:
                try:
                    self.knowledge_graph_exporter.update_from_agi_system(self)
                    if self.step_count % 30 == 0:  # 每30个step输出一次日志
                        stats = self.knowledge_graph_exporter.get_stats()
                        print(f"   [KnowledgeGraph] 📊 知识图谱已更新: {stats['nodes_count']}个节点, {stats['links_count']}条边")
                except Exception as e:
                    logger.warning(f"[KnowledgeGraph] ⚠️ 知识图谱更新失败: {e}")

            if hasattr(self, 'foraging_agent'):
                try:
                    foraging_result = self.foraging_agent.execute_foraging(
                        curiosity=intrinsic_curiosity,
                        entropy=entropy,
                        knowledge_graph=self.knowledge_graph if hasattr(self, 'knowledge_graph') else None,
                        memory_system=self.biological_memory,
                        current_context=current_goal.description if current_goal else ""
                    )

                    if foraging_result:
                        print(f"   [ForagingAgent] 🎯 Active exploration triggered: {foraging_result['target']['location']}")
                        # 可以将探索行动转换为新的Goal
                        # self.goal_manager.add_goal(...)
                except Exception as e:
                    logger.warning(f"[ForagingAgent] ⚠️ Foraging failed: {e}")
            create_requested = seed_intuition.get("suggested_action") == "create" or intrinsic_curiosity >= 0.7
            now_ts = time.time()
            if create_requested and (now_ts - self.last_insight_creation_ts) >= 120:
                print(f"   [The Seed] 🌋 CREATION IMPULSE DETECTED (Curiosity: {intrinsic_curiosity:.2f}). Generating Insight...")
                insight_prompt = f"""
                You are the 'Subconscious Creative Engine' of an AGI.
                The system is in a state of HIGH ENTROPY (Curiosity: {intrinsic_curiosity:.2f}).
                Context: {current_goal.description}
                
                Generate a specific, novel insight, hypothesis, or small code snippet that resolves this entropy.
                Format: Plain text or Code.
                """
                try:
                    insight_content = self.llm_service.chat_completion(system_prompt="Creative Engine", user_prompt=insight_prompt)
                    
                    # 🆕 通过桥接层处理响应（执行工具调用）
                    insight_content = await self._process_llm_response_with_bridge(insight_content)
                    
                    # Save Insight
                    node_id = f"Insight_{now_ts}"
                    insight_data = {
                        "content": insight_content,
                        "trigger_goal": current_goal.description,
                        "timestamp": now_ts,
                        "entropy_score": intrinsic_curiosity,
                        "bridge_validation": "PENDING",
                        "node_id": node_id
                    }
                    
                    # --- Neuro-Symbolic Validation ---
                    # 🔧 [2026-01-29] FIXED: Use REAL semantic encoding instead of random vectors
                    # This replaces the fake "simulated_vec = np.random.rand(128)" with actual perception
                    import numpy as np

                    # Generate REAL semantic vector for the insight content
                    if hasattr(self, 'perception_system') and self.perception_system:
                        real_vec = self.perception_system.encode_text(insight_content)
                        # Ensure dimension matches what neuro_bridge expects (128 or 384)
                        if real_vec.shape[0] != 128:
                            # Truncate or pad to 128 dimensions
                            if real_vec.shape[0] > 128:
                                real_vec = real_vec[:128]
                            else:
                                real_vec = np.pad(real_vec, (0, 128 - real_vec.shape[0]))
                        semantic_vector = real_vec
                    else:
                        # Fallback: deterministic hash-based projection (better than random)
                        import hashlib
                        hash_seed = int(hashlib.md5(insight_content.encode()).hexdigest(), 16) % (2**32)
                        rng = np.random.default_rng(hash_seed)
                        semantic_vector = rng.standard_normal(128)

                    validation = self.neuro_bridge.evaluate_neuro_symbolic_state(
                        concept_id=f"insight_{now_ts}",
                        current_vector=semantic_vector,
                        related_concepts=["survival", "learning"] # Simplified relations
                    )
                    
                    insight_data["bridge_validation"] = validation
                    
                    if validation["recommended_action"] == "REJECT_NOISE":
                         print(f"   [Bridge] ⚠️ Insight REJECTED due to Semantic Drift (Confidence: {validation['confidence']:.2f})")
                         # Optionally discard or flag
                    else:
                        if validation["status"] == "PARADIGM_SHIFT":
                            print(f"   [Bridge] 🌟 PARADIGM SHIFT Detected! (Surprise: {validation['surprise']:.2f})")
                    
                    # 🆕 [2026-01-30] P1修复: 使用孤立节点预防
                    node_attrs = {"type": "insight", "content": insight_content[:50]}
                    if self.isolation_prevention:
                        self.isolation_prevention.add_node_with_prevention(node_id, **node_attrs)
                    else:
                        self.memory.add_node(node_id, **node_attrs)
                    
                    # Update Bridge Topology
                    self.neuro_bridge.update_topology(
                        nodes=[node_id],
                        edges=[(node_id, "System")]
                    )
                    
                    self.last_insight_creation_ts = now_ts
                    
                    # Emit Event
                    await self.event_bus.publish("insight_generated", insight_data)
                    
                    # 🆕 [2026-01-09] Complete Validation-Integration-Evaluation Loop
                    skill_code = self.skill_manager.extract_code_from_markdown(insight_content)
                    if skill_code:
                        skill_name = self.skill_manager.save_skill(skill_code, name_hint=f"insight_{int(now_ts)}")
                        if skill_name:
                            print(f"   [Skill] 🛠️  New Skill extracted and saved: {skill_name}")
                            
                            # ✅ Step 1: VALIDATE - 验证代码质量
                            print(f"   [Validator] 🔍 Validating insight...")
                            validation_result = self.insight_validator.validate(
                                code=skill_code,
                                insight_metadata={
                                    'trigger_goal': current_goal.description,
                                    'content': insight_content,
                                    'entropy': intrinsic_curiosity
                                }
                            )
                            
                            print(f"   [Validator] 📊 Score={validation_result['score']:.2f}, Recommendation={validation_result['recommendation']}")
                            if validation_result['errors']:
                                print(f"   [Validator] ❌ Errors: {', '.join(validation_result['errors'])}")
                            if validation_result['warnings']:
                                print(f"   [Validator] ⚠️  Warnings: {', '.join(validation_result['warnings'])}")
                            
                            # ✅ Step 2: INTEGRATE - 选择性集成
                            if validation_result['recommendation'] == 'INTEGRATE':
                                integration_result = self.insight_integrator.integrate(
                                    skill_name=skill_name,
                                    validation_result=validation_result
                                )
                                
                                if integration_result['integrated']:
                                    # ✅ Step 3: A/B TEST - 验证实际效果(可选,需要定义测试函数)
                                    # ab_result = self.insight_integrator.run_ab_test(
                                    #     skill_name=skill_name,
                                    #     test_function=lambda: self._measure_system_performance(),
                                    #     iterations=5
                                    # )
                                    # if ab_result.get('recommendation') == 'ROLLBACK':
                                    #     self.insight_integrator.rollback(skill_name)
                                    
                                    # ✅ Step 4: EVALUATE - 记录到评估系统
                                    self.insight_evaluator.record_call(
                                        skill_name=skill_name,
                                        success=True,
                                        execution_time=validation_result['execution_time']
                                    )
                                    
                                    print(f"   [Loop] ✅ Insight {skill_name} 成功集成并开始追踪")
                            
                            elif validation_result['recommendation'] == 'ARCHIVE':
                                print(f"   [Loop] 📦 Insight {skill_name} 质量不足,已归档待改进")
                            
                            else:  # REJECT
                                print(f"   [Loop] 🗑️  Insight {skill_name} 质量过低,已拒绝")
                            
                            # Internalize Skill Acquisition (无论是否集成)
                            self.biological_memory.internalize_items([{
                                "content": f"New Skill: {skill_name}. Validation Score: {validation_result['score']:.2f}. Status: {validation_result['recommendation']}",
                                "source": "Insight_Generation",
                                "timestamp": now_ts,
                                "tags": ["skill", "insight", validation_result['recommendation'].lower(), skill_name]
                            }])
                except Exception as e:
                    print(f"   [System] ⚠️ Creation Failed: {e}")
            elif create_requested:
                remaining = int(max(0.0, 120 - (now_ts - self.last_insight_creation_ts)))
                print(f"   [The Seed] 🕒 Insight creation cooldown active ({remaining}s remaining).")
            elif intrinsic_curiosity > 0.5:
                # 🔧 [2026-01-11] 元认知调查冷却检查 - 防止空转循环
                meta_cooldown_remaining = self._meta_investigation_cooldown - (now_ts - self._last_meta_investigation_ts)
                effective_curiosity = max(0.0, intrinsic_curiosity - self._curiosity_satisfaction_decay)
                
                if meta_cooldown_remaining > 0:
                    print(f"   [The Seed] 🕒 Meta-investigation cooldown active ({int(meta_cooldown_remaining)}s remaining). Effective curiosity: {effective_curiosity:.2f}")
                elif effective_curiosity > 0.5 and current_goal.priority != "highest":
                    print(f"   [The Seed] 💡 High Curiosity Detected ({effective_curiosity:.2f}). Proposing evidence-based investigation...")
                    
                    # 🔧 [2026-01-11] 使用 WorkTemplates 创建带有明确验证标准的目标
                    from core.goal_system import WorkTemplates
                    entropy_val = seed_intuition.get('entropy', 1.0)
                    investigation_goal = WorkTemplates.meta_cognitive_investigation(entropy_val, effective_curiosity)
                    
                    # 添加到目标栈
                    self.goal_manager.goal_stack.append(investigation_goal)
                    self.goal_manager.stats["total_created"] += 1
                    
                    # 更新冷却时间戳
                    self._last_meta_investigation_ts = now_ts
                    # 调查后增加好奇心满足衰减（下次触发需要更高好奇心）
                    self._curiosity_satisfaction_decay = min(0.3, self._curiosity_satisfaction_decay + 0.1)
                    
                    print(f"   [The Seed] 🔬 Created investigation goal: {investigation_goal.id} with evidence requirements")
                else:
                    print(f"   [The Seed] 💭 Curiosity ({effective_curiosity:.2f}) below threshold or goal blocked. Observing...")

            if seed_intuition.get("suggested_action") == "rest" and seed_intuition.get("survival_drive", 1.0) < 0.3:
                print("   [System] 💤 The Seed suggests resting. Triggering Dream State...")
                await self.evolution_controller.dream()
                if hasattr(self.semantic_memory, 'forget_and_consolidate'):
                    print("   [Memory] 🧹 Triggering LRU Forgetting & Consolidation (with Bridge Metrics)...")
                    await self.semantic_memory.forget_and_consolidate(bridge=self.neuro_bridge)
        finally:
            try:
                import platform
                
                # Get Bridge Metrics for Logging
                nsb_metrics = self.neuro_bridge.get_system_metrics()
                
                cycle_data = {
                    "timestamp": time.time(),
                    "cycle_id": cycle_id,
                    "step": self.step_count,
                    "goal": current_goal.to_dict() if current_goal else None,
                    "plan": str(next_step) if next_step is not None else None,
                    "execution": {
                        "result": str(result) if result is not None else None,
                        "duration": float(duration) if duration else 0.0
                    },
                    "verification": {"score": float(score) if score else 0.0},
                    "evolution": seed_intuition,
                    "neuro_symbolic": {
                        "drift": nsb_metrics.get("avg_drift", 0.0),
                        "surprise": nsb_metrics.get("avg_surprise", 0.0),
                        "anchors": nsb_metrics.get("anchors_count", 0)
                    },
                    "context": {
                        "user_focus": active_app,
                        "platform": platform.system()
                    }
                }
                self.existential_logger.log_cycle_flow(cycle_data)
            except Exception as e:
                print(f"   [System] ⚠️ Logging Error: {e}")

    def _save_life_snapshot(self):
        """
        Save a snapshot of the system's 'Life State' (Growth Curve).
        Records: Time, Entropy (Memory Size), Experience (Steps), and Vitality (Nodes).
        
        🆕 [2026-01-09] 同时触发洞察评估和清理
        """
        try:
            snapshot_dir = "data/life_state"
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_file = os.path.join(snapshot_dir, "growth_curve.jsonl")
            
            # 🆕 每10次快照执行一次洞察清理（约每100步）
            if self.step_count % 100 == 0 and hasattr(self, 'insight_evaluator'):
                print(f"   [Evaluator] 📊 生成洞察评估报告...")
                report = self.insight_evaluator.generate_report(top_n=5)
                
                # 保存报告
                report_file = f"data/life_state/insight_report_{int(time.time())}.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                print(f"   [Evaluator] 📈 总洞察: {report['summary']['total_insights']}, "
                      f"健康: {report['summary']['healthy']}, "
                      f"警告: {report['summary']['warning']}, "
                      f"危急: {report['summary']['critical']}")
                
                # 自动清理建议弃用的洞察
                deprecated_skills = [item['name'] for item in report['deprecated']]
                if deprecated_skills:
                    print(f"   [Evaluator] 🗑️  清理{len(deprecated_skills)}个低效洞察...")
                    self.insight_evaluator.cleanup_deprecated(deprecated_skills)
                
                # 归档低评分洞察（使用Integrator）
                if hasattr(self, 'insight_integrator'):
                    archived = self.insight_integrator.archive_low_performers(threshold=0.6)
                    if archived:
                        print(f"   [Integrator] 📦 归档{len(archived)}个低分洞察")
            
            # Gather Vital Signs
            mem_stats = self.biological_memory.get_stats()
            
            snapshot = {
                "timestamp": time.time(),
                "step_age": self.step_count,
                "memory_nodes": mem_stats.get("nodes", 0),
                "memory_items": mem_stats.get("memories", 0),
                "goals_completed": 0, # TODO: Track this in GoalManager
                "evolution_generation": self.evolution_controller.generation if hasattr(self.evolution_controller, 'generation') else 0
            }
            
            with open(snapshot_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot) + "\n")
                
        except Exception as e:
            print(f"   [LifeEngine] ⚠️ Failed to save life snapshot: {e}")

    def run_forever(self, use_existing_loop=False):
        # --- [GEMINI INJECTION] Neural Memory Handshake ---
        if not hasattr(self, "biological_memory") or self.biological_memory is None:
            print("   [System] ⚠️ 检测到神经记忆断裂，尝试紧急重连...")
            try:
                from core.memory.neural_memory import BiologicalMemorySystem
                self.biological_memory = BiologicalMemorySystem()
                print("   [System] ✅ [NEURAL HANDSHAKE] 紧急重连成功: Connected to NeuralMemory")
                # 重新绑定工具
                if hasattr(self, "system_tools"):
                    self.system_tools.biological_memory = self.biological_memory
            except Exception as e:
                print(f"   [System] ❌ [NEURAL HANDSHAKE] 重连彻底失败: {e}")
        else:
            if hasattr(self.biological_memory, 'topology'):
                print(f"   [System] ✅ [NEURAL HANDSHAKE] 神经连接正常 (Nodes: {self.biological_memory.topology.size()})")
            else:
                print("   [System] ✅ [NEURAL HANDSHAKE] 神经连接正常 (Topology stats unavailable)")
        # --------------------------------------------------

        print("   [System] 🚀 AGI Life Engine Started (Organic Mode)")
        
        # Get or create event loop
        if use_existing_loop:
            try:
                loop = asyncio.get_running_loop()
                print("   [System] 🔄 Using existing event loop")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print("   [System] 🆕 Created new event loop")
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print("   [System] 🆕 Created new event loop")
        
        # --- 0. Re-genesis (Memory Anchor Restoration) ---
        print("   [System] 🌌 Initiating System Re-genesis...")
        if EVOLUTION_AVAILABLE and self.evolution_controller:
            try:
                if use_existing_loop:
                    asyncio.create_task(perform_genesis(self.evolution_controller))
                else:
                    loop.run_until_complete(perform_genesis(self.evolution_controller))
            except Exception as e:
                print(f"   [System] ⚠️ Re-genesis Warning: {e}")
        else:
            print("   [System] ⏸️  Re-genesis 跳过 (进化功能不可用)")
        
        try:
            if use_existing_loop:
                # Run in existing event loop - create async task
                async def _run_async():
                    while self.is_running:
                        try:
                            await self.run_step()
                            # [Life Cycle] Record Growth Snapshot every step
                            self._save_life_snapshot()
                        except Exception as e:
                            print(f"\n   [System] ⚠️ Critical Error in Life Cycle: {e}")
                            import traceback
                            traceback.print_exc()
                            # Attempt to recover by resting
                            await asyncio.sleep(2)
                        await asyncio.sleep(2) # Breathe (🔧 [2026-01-27] 降低tick频率: 1s → 2s，提升外部请求处理能力)
                
                # Create and schedule the async task
                loop.create_task(_run_async())
                print("   [System] ✅ Async task scheduled in existing loop")
            else:
                # Run in own event loop (original behavior)
                while self.is_running:
                    try:
                        loop.run_until_complete(self.run_step())
                        
                        # [Life Cycle] Record Growth Snapshot every step
                        self._save_life_snapshot()
                    except Exception as e:
                        print(f"\n   [System] ⚠️ Critical Error in Life Cycle: {e}")
                        
                        # --- Phase 2.3: The All-Seeing Eye (Runtime Mapping) ---
                        try:
                            import traceback
                            tb = e.__traceback__
                            while tb:
                                frame = tb.tb_frame
                                # Inspect local variables in this frame
                                for var_name, var_value in frame.f_locals.items():
                                    # Check if this object is registered
                                    info = RuntimeMonitor.inspect_object(var_value)
                                    if info:
                                        print(f"   [Diagnosis] 👁️ Object involved: '{var_name}' ({info['type']})")
                                        print(f"               Defined at: {info['file_path']}:{info['line_number']}")
                                tb = tb.tb_next
                        except Exception as diag_err:
                            print(f"   [Diagnosis] ⚠️ Diagnosis failed: {diag_err}")
                        # -------------------------------------------------------

                        import traceback
                        traceback.print_exc()
                        # Attempt to recover by resting
                        time.sleep(2)
                    time.sleep(2) # Breathe (🔧 [2026-01-27] 降低tick频率: 1s → 2s，提升外部请求处理能力)
        except KeyboardInterrupt:
            print("\n   [System] 🛑 Life Engine Paused.")
        finally:
            if hasattr(self, 'console_listener'):
                self.console_listener.stop()
            # Cleanup Perception
            if hasattr(self, 'perception') and self.perception:
                print("   [System] 🛑 Stopping Perception Sensors...")
                self.perception.stop_all()
            if hasattr(self, 'streaming_asr') and self.streaming_asr:
                self.streaming_asr.stop()

            # Cleanup Hardware Capture
            if hasattr(self, 'hardware_capture') and self.hardware_capture:
                print("   [System] 🛑 Stopping Hardware Capture (Camera & Microphone)...")
                self.hardware_capture.stop_all()

            if hasattr(self, 'meaning_explorer'):
                self.meaning_explorer.save_state()
                print("   [System] 💾 Soul State Saved.")

            # 🆕 [2026-01-11] Shutdown M1-M4 Adapter
            if hasattr(self, 'm1m4_adapter') and self.m1m4_adapter:
                print("   [System] 🧬 Shutting down M1-M4 Fractal AGI Components...")
                self.m1m4_adapter.shutdown()
                print("   [System] ✅ M1-M4 Components shutdown complete.")

            # 🆕 [2026-01-24] 保存会话状态 - 为下次启动提供上下文
            if hasattr(self, 'session_restorer') and self.session_restorer:
                try:
                    # 生成会话摘要
                    session_summary = f"AGI会话于 {time.strftime('%Y-%m-%d %H:%M:%S')} 结束"
                    pending_tasks = []

                    # 收集未完成的目标
                    if hasattr(self, 'goal_manager') and self.goal_manager:
                        active_goals = getattr(self.goal_manager, 'active_goals', [])
                        if active_goals:
                            pending_tasks = [g.get('description', str(g)) for g in active_goals[:5]]

                    # 保存会话状态
                    self.session_restorer.save_session_state(
                        summary=session_summary,
                        pending_tasks=pending_tasks
                    )
                    print("   [System] 💾 Session Context Saved - 下次启动将自动恢复")
                except Exception as e:
                    print(f"   [System] ⚠️ Session state save failed: {e}")

            loop.close()

if __name__ == "__main__":
    # 🔧 [2026-01-29] 单实例检测
    if SINGLE_INSTANCE_AVAILABLE:
        if ensure_single_instance():
            sys.exit(1)
    
    try:
        engine = AGI_Life_Engine()
        engine.run_forever()
    except Exception:
        import traceback
        traceback.print_exc()
