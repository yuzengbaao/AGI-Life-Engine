"""
洞察评估器 (Insight Evaluator)
持续跟踪已集成洞察的实际效果,提供反馈循环。

评估维度:
1. 使用频率 - 洞察被调用的次数
2. 性能影响 - 对系统整体性能的贡献
3. 错误率 - 洞察执行失败的比例
4. 价值衰减 - 洞察效果随时间的变化
5. 依赖健康度 - 其他组件对该洞察的依赖程度
"""

import time
import json
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

class InsightEvaluator:
    """洞察评估器 - 持续监控洞察的实际价值"""

    def __init__(self, metrics_file: str = "data/skills/metrics.json", event_callback=None):
        """
        初始化洞察评估器

        Args:
            metrics_file: 指标存储文件路径
            event_callback: 事件回调函数(可选)，签名 callback(event_type: str, data: dict)，可同步或异步
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # 🆕 [2026-01-10] 事件发布回调（用于拓扑图中 InsightEvaluator → Engine 的事件回流）
        self._event_callback = event_callback

        # 加载历史指标
        self.metrics = self._load_metrics()

        # 实时指标缓存
        self.session_metrics = defaultdict(lambda: {
            'calls': 0,
            'successes': 0,
            'failures': 0,
            'total_time': 0.0,
            'errors': []
        })

    def set_event_callback(self, callback):
        """设置事件回调函数（用于运行时注入）"""
        self._event_callback = callback
    
    def _load_metrics(self) -> Dict:
        """加载历史指标"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {'insights': {}, 'sessions': []}
        return {'insights': {}, 'sessions': []}
    
    def _save_metrics(self):
        """保存指标"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Evaluator] ⚠️ 保存指标失败: {e}")
    
    def record_call(self, 
                    skill_name: str, 
                    success: bool, 
                    execution_time: float,
                    error: str = None):
        """记录单次调用"""
        # 更新会话缓存
        session = self.session_metrics[skill_name]
        session['calls'] += 1
        session['total_time'] += execution_time
        
        if success:
            session['successes'] += 1
        else:
            session['failures'] += 1
            if error:
                session['errors'].append({
                    'error': error,
                    'timestamp': time.time()
                })
        
        # 更新持久化指标
        if skill_name not in self.metrics['insights']:
            self.metrics['insights'][skill_name] = {
                'total_calls': 0,
                'total_successes': 0,
                'total_failures': 0,
                'total_time': 0.0,
                'first_used': time.time(),
                'last_used': time.time(),
                'daily_calls': [],
                'performance_history': []
            }
        
        insight_metrics = self.metrics['insights'][skill_name]
        insight_metrics['total_calls'] += 1
        insight_metrics['total_time'] += execution_time
        insight_metrics['last_used'] = time.time()
        
        if success:
            insight_metrics['total_successes'] += 1
        else:
            insight_metrics['total_failures'] += 1
        
        # 记录每日调用(用于检测衰减)
        today = datetime.now().date().isoformat()
        daily_calls = insight_metrics.setdefault('daily_calls', [])
        if not daily_calls or daily_calls[-1]['date'] != today:
            daily_calls.append({'date': today, 'count': 1})
        else:
            daily_calls[-1]['count'] += 1
        
        # 保持最近30天数据
        if len(daily_calls) > 30:
            insight_metrics['daily_calls'] = daily_calls[-30:]
    
    def evaluate(self, skill_name: str) -> Dict[str, Any]:
        """
        评估单个洞察
        
        返回格式:
        {
            'score': float (0-1),
            'usage_frequency': float,
            'success_rate': float,
            'avg_execution_time': float,
            'value_decay': float,  # 负数表示衰减
            'recommendation': str,  # 'KEEP', 'IMPROVE', 'DEPRECATE'
            'health': str  # 'HEALTHY', 'WARNING', 'CRITICAL'
        }
        """
        if skill_name not in self.metrics['insights']:
            return {
                'score': 0.0,
                'usage_frequency': 0.0,
                'success_rate': 0.0,
                'recommendation': 'NEW',
                'health': 'UNKNOWN'
            }
        
        m = self.metrics['insights'][skill_name]
        
        # 1. 使用频率得分 (0-1)
        calls_per_day = self._calculate_calls_per_day(m)
        freq_score = min(1.0, calls_per_day / 10.0)  # 每天10次调用=满分
        
        # 2. 成功率得分 (0-1)
        success_rate = m['total_successes'] / m['total_calls'] if m['total_calls'] > 0 else 0
        
        # 3. 性能得分 (0-1, 执行时间越短越好)
        avg_time = m['total_time'] / m['total_calls'] if m['total_calls'] > 0 else 0
        perf_score = max(0, 1.0 - avg_time / 0.5)  # 0.5s内完成=满分
        
        # 4. 价值衰减检测
        decay = self._calculate_value_decay(m)
        
        # 综合评分
        weights = {'frequency': 0.3, 'success': 0.4, 'performance': 0.2, 'decay': 0.1}
        score = (
            weights['frequency'] * freq_score +
            weights['success'] * success_rate +
            weights['performance'] * perf_score +
            weights['decay'] * (1.0 + decay)  # decay为负,所以加上它
        )
        
        # 健康状态
        if success_rate > 0.9 and decay > -0.1:
            health = 'HEALTHY'
        elif success_rate > 0.7 and decay > -0.3:
            health = 'WARNING'
        else:
            health = 'CRITICAL'
        
        # 建议
        if score > 0.7 and health == 'HEALTHY':
            recommendation = 'KEEP'
        elif score > 0.5:
            recommendation = 'IMPROVE'
        else:
            recommendation = 'DEPRECATE'
        
        return {
            'score': score,
            'usage_frequency': calls_per_day,
            'success_rate': success_rate,
            'avg_execution_time': avg_time,
            'value_decay': decay,
            'recommendation': recommendation,
            'health': health,
            'total_calls': m['total_calls'],
            'days_active': (time.time() - m['first_used']) / 86400
        }
    
    def _calculate_calls_per_day(self, metrics: Dict) -> float:
        """计算每日平均调用次数"""
        daily_calls = metrics.get('daily_calls', [])
        if not daily_calls:
            return 0.0
        
        # 最近7天平均
        recent_calls = daily_calls[-7:]
        return sum(d['count'] for d in recent_calls) / len(recent_calls)
    
    def _calculate_value_decay(self, metrics: Dict) -> float:
        """
        计算价值衰减率
        比较最近7天与之前7天的调用频率
        
        返回: -1.0 到 1.0 (负数=衰减, 正数=增长)
        """
        daily_calls = metrics.get('daily_calls', [])
        if len(daily_calls) < 14:
            return 0.0  # 数据不足
        
        recent_7 = daily_calls[-7:]
        previous_7 = daily_calls[-14:-7]
        
        recent_avg = sum(d['count'] for d in recent_7) / 7
        previous_avg = sum(d['count'] for d in previous_7) / 7
        
        if previous_avg == 0:
            return 0.0
        
        decay = (recent_avg - previous_avg) / previous_avg
        return max(-1.0, min(1.0, decay))  # 限制在[-1, 1]
    
    def generate_report(self, top_n: int = 10, emit_event: bool = True) -> Dict[str, Any]:
        """
        生成评估报告
        
        包含:
        1. Top N 最有价值洞察
        2. 需要改进的洞察
        3. 建议弃用的洞察
        4. 总体统计
        
        Args:
            top_n: 返回Top N个最佳洞察
            emit_event: 是否发布评估完成事件（用于拓扑图中的事件回流）
        """
        # 评估所有洞察
        evaluations = {}
        for skill_name in self.metrics['insights'].keys():
            evaluations[skill_name] = self.evaluate(skill_name)
        
        # 排序
        sorted_by_score = sorted(
            evaluations.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        # 分类
        top_performers = sorted_by_score[:top_n]
        need_improvement = [
            (name, eval) for name, eval in evaluations.items()
            if eval['recommendation'] == 'IMPROVE'
        ]
        deprecated = [
            (name, eval) for name, eval in evaluations.items()
            if eval['recommendation'] == 'DEPRECATE'
        ]
        
        # 总体统计
        total_insights = len(evaluations)
        healthy_count = sum(1 for e in evaluations.values() if e['health'] == 'HEALTHY')
        warning_count = sum(1 for e in evaluations.values() if e['health'] == 'WARNING')
        critical_count = sum(1 for e in evaluations.values() if e['health'] == 'CRITICAL')
        
        avg_score = sum(e['score'] for e in evaluations.values()) / total_insights if total_insights > 0 else 0
        avg_success_rate = sum(e['success_rate'] for e in evaluations.values()) / total_insights if total_insights > 0 else 0
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_insights': total_insights,
                'healthy': healthy_count,
                'warning': warning_count,
                'critical': critical_count,
                'average_score': avg_score,
                'average_success_rate': avg_success_rate
            },
            'top_performers': [
                {
                    'name': name,
                    **eval_data
                }
                for name, eval_data in top_performers
            ],
            'need_improvement': [
                {
                    'name': name,
                    'score': eval_data['score'],
                    'success_rate': eval_data['success_rate'],
                    'issue': self._diagnose_issue(eval_data)
                }
                for name, eval_data in need_improvement
            ],
            'deprecated': [
                {
                    'name': name,
                    'score': eval_data['score'],
                    'reason': self._deprecation_reason(eval_data)
                }
                for name, eval_data in deprecated
            ]
        }
        
        # 🆕 [2026-01-10] 发布评估完成事件（修复拓扑图中InsightEvaluator→Engine的event连接）
        if emit_event and self._event_callback:
            try:
                import asyncio
                event_data = {
                    'report_summary': report['summary'],
                    'top_performer': top_performers[0][0] if top_performers else None,
                    'critical_count': critical_count,
                    'deprecated_count': len(deprecated)
                }
                # 支持同步和异步回调
                if asyncio.iscoroutinefunction(self._event_callback):
                    asyncio.create_task(self._event_callback('insight_evaluation_complete', event_data))
                else:
                    self._event_callback('insight_evaluation_complete', event_data)
            except Exception as e:
                print(f"[Evaluator] ⚠️ 事件发布失败: {e}")
        
        return report
    
    def _diagnose_issue(self, eval_data: Dict) -> str:
        """诊断洞察的问题"""
        issues = []
        
        if eval_data['success_rate'] < 0.7:
            issues.append(f"低成功率({eval_data['success_rate']:.1%})")
        
        if eval_data['usage_frequency'] < 1.0:
            issues.append(f"低使用率({eval_data['usage_frequency']:.1f}次/天)")
        
        if eval_data['value_decay'] < -0.3:
            issues.append(f"价值衰减({eval_data['value_decay']:+.1%})")
        
        if eval_data['avg_execution_time'] > 0.5:
            issues.append(f"性能慢({eval_data['avg_execution_time']:.2f}s)")
        
        return ', '.join(issues) if issues else '未知问题'
    
    def _deprecation_reason(self, eval_data: Dict) -> str:
        """弃用原因"""
        if eval_data['success_rate'] < 0.5:
            return f"高失败率({eval_data['success_rate']:.1%})"
        elif eval_data['usage_frequency'] < 0.1:
            return f"极少使用({eval_data['usage_frequency']:.2f}次/天)"
        elif eval_data['value_decay'] < -0.5:
            return f"严重衰减({eval_data['value_decay']:+.1%})"
        else:
            return f"综合评分过低({eval_data['score']:.2f})"
    
    def cleanup_deprecated(self, skill_names: List[str], archive_dir: str = "data/skills/deprecated"):
        """清理弃用的洞察"""
        from pathlib import Path
        import shutil
        
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        cleaned = []
        for skill_name in skill_names:
            try:
                # 移动文件到deprecated目录
                skill_file = Path(f"data/skills/{skill_name}.py")
                if skill_file.exists():
                    shutil.move(str(skill_file), str(archive_path / skill_file.name))
                    cleaned.append(skill_name)
                    
                    # 标记为已弃用
                    if skill_name in self.metrics['insights']:
                        self.metrics['insights'][skill_name]['deprecated'] = True
                        self.metrics['insights'][skill_name]['deprecated_at'] = time.time()
                    
                    print(f"   [Evaluator] 🗑️ 清理弃用洞察: {skill_name}")
            except Exception as e:
                print(f"   [Evaluator] ⚠️ 清理失败 {skill_name}: {e}")
        
        if cleaned:
            self._save_metrics()
        
        return cleaned
    
    def end_session(self):
        """结束评估会话,保存所有指标"""
        # 记录会话摘要
        session_summary = {
            'timestamp': time.time(),
            'insights_called': len(self.session_metrics),
            'total_calls': sum(m['calls'] for m in self.session_metrics.values()),
            'total_successes': sum(m['successes'] for m in self.session_metrics.values()),
            'total_failures': sum(m['failures'] for m in self.session_metrics.values())
        }
        
        self.metrics['sessions'].append(session_summary)
        
        # 保持最近100个会话
        if len(self.metrics['sessions']) > 100:
            self.metrics['sessions'] = self.metrics['sessions'][-100:]
        
        self._save_metrics()
        
        # 清空会话缓存
        self.session_metrics.clear()
