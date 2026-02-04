import time
import logging
import sys # Added import sys
from enum import Enum, auto
from typing import Dict, Any

from core.monitor import ProprioceptiveMonitor
from core.checkpoint import CheckpointManager
from core.memory import ExperienceMemory
from core.learning import MetricDrivenEvolver
from core.logger import ThoughtLogger
from core.knowledge_graph import ArchitectureKnowledgeGraph
from core.agents_legacy import AgentOrchestrator

class State(Enum):
    INIT = auto()
    THINKING = auto()
    CODING = auto()
    MAINTENANCE = auto()
    SLEEPING = auto()

class LearningStateMachine:
    """
    状态机引擎 (State Machine Engine) - Phase 3 Co-Evolution
    集成知识图谱 (Knowledge Graph) 与 智能体 (Agents)
    """
    def __init__(self):
        # 1. 基础生存组件
        self.monitor = ProprioceptiveMonitor()
        self.checkpoint_mgr = CheckpointManager()
        self.logger = logging.getLogger("Engine")
        self._setup_logging()
        
        # 2. 认知增强组件 (Phase 2)
        self.memory = ExperienceMemory()
        self.evolver = MetricDrivenEvolver()
        self.thought_logger = ThoughtLogger()
        
        # 3. 协同进化组件 (Phase 3)
        self.knowledge_graph = ArchitectureKnowledgeGraph()
        self.agents = AgentOrchestrator()
        
        # 4. 核心上下文
        self.context = {
            'iteration': 0,
            'current_state': State.INIT.name,
            'last_active': time.time(),
            'current_strategy': 'Standard',
            'latest_metrics': {}
        }
        
        # 5. 尝试恢复状态
        self._restore_state()

    def _setup_logging(self):
        # Prevent adding duplicate handlers if re-initialized
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout) # Ensure writing to stdout
            formatter = logging.Formatter('%(asctime)s - ENGINE - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _restore_state(self):
        saved_state = self.checkpoint_mgr.load_latest()
        if saved_state:
            self.context.update(saved_state)
            self.logger.info(f"Restored state: Iteration {self.context['iteration']}, State {self.context['current_state']}")
        else:
            self.logger.info("Initialized new state.")

    def run(self, max_iterations=None):
        """主循环"""
        self.logger.info("Engine started (Phase 3: Co-Evolution).")
        
        try:
            while True:
                # 0. 检查退出条件
                if max_iterations and self.context['iteration'] >= max_iterations:
                    self.logger.info("Max iterations reached. Stopping.")
                    break

                # 1. 资源健康检查
                health = self.monitor.check_health()
                action = self.monitor.suggest_action(health)
                
                # 更新当前指标
                self.context['latest_metrics'] = self.monitor.get_resource_usage()
                
                if action == "EMERGENCY_SAVE_AND_EXIT":
                    self.logger.critical("Emergency shutdown triggered!")
                    self.checkpoint_mgr.save_checkpoint(self.context)
                    break
                elif action == "GC_AND_THROTTLE":
                    self.logger.warning("Throttling...")
                    time.sleep(2) 
                
                # 2. 执行状态逻辑
                self._step()
                
                # 3. 定期保存 (每5步)
                if self.context['iteration'] % 5 == 0:
                    self.checkpoint_mgr.save_checkpoint(self.context)
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            self.logger.info("User interrupted. Saving state...")
            self.checkpoint_mgr.save_checkpoint(self.context)
        except Exception as e:
            self.logger.error(f"Unexpected crash: {e}")
            self.checkpoint_mgr.save_checkpoint(self.context)
            raise

    def _step(self):
        current_state_name = self.context['current_state']
        
        if current_state_name == State.INIT.name:
            self._do_init()
            self.context['current_state'] = State.THINKING.name
            
        elif current_state_name == State.THINKING.name:
            self._do_thinking()
            self.context['current_state'] = State.CODING.name
            
        elif current_state_name == State.CODING.name:
            self._do_coding()
            self.context['current_state'] = State.MAINTENANCE.name
            
        elif current_state_name == State.MAINTENANCE.name:
            self._do_maintenance()
            self.context['iteration'] += 1
            self.context['current_state'] = State.THINKING.name

    # --- 具体行为模拟 (Phase 3 Enhanced) ---
    def _do_init(self):
        self.logger.info("Initializing subsystems (Graph, Agents)...")
        stats = self.knowledge_graph.get_stats()
        self.logger.info(f"Knowledge Graph Loaded: {stats['node_count']} nodes.")
        self.thought_logger.log_thought("INIT", {"status": "Phase 3 Ready", "graph_stats": stats})

    def _do_thinking(self):
        # 1. 分析
        metrics = self.context.get('latest_metrics', {}).copy()
        last_result = self.context.get('last_action_result', {})
        metrics['task_success_rate'] = last_result.get('success_score', 0.0)
        
        analysis = self.evolver.analyze_metrics(metrics)
        
        # 2. 决策与查询图谱
        strategy = self.context['current_strategy']
        
        # 如果当前策略停滞，查询知识图谱寻找最佳实践
        best_practice = None
        if analysis['status'] == "STAGNANT":
            practices = self.knowledge_graph.query_best_practice("strategy")
            if practices:
                best_practice = practices[0]
                strategy = f"AdoptedBestPractice({best_practice['decision']})"
                self.logger.info(f"💡 Found best practice from graph: {strategy}")
            else:
                mutation = self.evolver.suggest_mutation(analysis['suggestion'])
                if mutation != "No mutation":
                    strategy = f"Mutated({mutation})"
        
        self.context['current_strategy'] = strategy
        
        # 3. 形成思维
        thought_content = {
            "metrics_analysis": analysis,
            "current_strategy": strategy,
            "best_practice_found": best_practice is not None,
            "goal": "Optimize Efficiency"
        }
        
        self.logger.info(f"[Iter {self.context['iteration']}] Thinking... Strategy: {strategy}")
        self.thought_logger.log_thought("THINKING", thought_content, meta={"iter": self.context['iteration']})
        self.context['last_thought'] = thought_content

    def _do_coding(self):
        strategy = self.context.get('last_thought', {}).get('current_strategy', 'Standard')
        self.logger.info(f"[Iter {self.context['iteration']}] Delegating to Agents (Strategy: {strategy})...")
        
        # 使用 AgentOrchestrator 进行开发
        task_context = {"strategy": strategy, "complexity": 1.2}
        agent_result = self.agents.run_development_cycle(task_context)
        
        success_score = agent_result['final_outcome']
        
        # 如果 Agent 审核通过，认为是一次成功的尝试，分数较高
        if success_score > 0.9:
            # 引入一点随机波动模拟运行时差异
            import random
            success_score = 0.95 + (random.random() * 0.05)
        else:
            success_score = 0.5
            
        self.context['last_action_result'] = {
            "action": "AgentDevCycle",
            "strategy": strategy,
            "success_score": success_score,
            "agent_details": agent_result
        }
        
        self.thought_logger.log_thought("CODING", self.context['last_action_result'], meta={"iter": self.context['iteration']})

    def _do_maintenance(self):
        result = self.context.get('last_action_result', {})
        thought = self.context.get('last_thought', {})
        
        # 1. 存入经验记忆 (Memory)
        self.memory.add_experience(
            context=f"Strategy_{result.get('strategy')}",
            action=result.get('action'),
            outcome=result.get('success_score', 0.0),
            details={"thought": thought}
        )
        
        # 2. 存入知识图谱 (Knowledge Graph) - 只存储高价值的成功案例
        if result.get('success_score', 0.0) > 0.9:
            node_id = self.knowledge_graph.add_decision_node(
                context=f"context_strategy_{result.get('strategy')}",
                decision=result.get('strategy'),
                outcome=result.get('success_score'),
                metadata={"agent_review": result.get('agent_details', {}).get('review')}
            )
            self.logger.info(f"✨ Archived successful decision to Knowledge Graph: {node_id}")

        self.logger.info(f"[Iter {self.context['iteration']}] Maintenance & Learning complete.")
        self.thought_logger.log_thought("MAINTENANCE", {"status": "Knowledge Consolidted"}, meta={"iter": self.context['iteration']})
