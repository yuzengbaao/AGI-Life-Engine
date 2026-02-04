"""
洞察集成器 (Insight Integrator)
负责将验证通过的洞察集成到系统中，实现A/B测试和版本管理。

集成策略:
1. 选择性集成 - 只集成score>0.8的洞察
2. A/B测试 - 对比集成前后的系统性能
3. 版本管理 - 支持回滚到之前的版本
4. 热加载 - 无需重启系统即可应用新洞察
5. 依赖管理 - 处理洞察之间的依赖关系
"""

import os
import json
import time
import shutil
import importlib
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

class InsightIntegrator:
    """洞察集成器 - 将有价值的洞察动态集成到系统中"""
    
    def __init__(self, 
                 skills_dir: str = "data/skills",
                 active_dir: str = "data/skills/active",
                 archive_dir: str = "data/skills/archive",
                 versions_file: str = "data/skills/versions.json"):
        
        self.skills_dir = Path(skills_dir)
        self.active_dir = Path(active_dir)
        self.archive_dir = Path(archive_dir)
        self.versions_file = Path(versions_file)
        
        # 确保目录存在
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载版本历史
        self.versions = self._load_versions()
        
        # 当前激活的洞察
        self.active_insights = {}
        
        # A/B测试结果
        self.ab_test_results = []
    
    def _load_versions(self) -> Dict:
        """加载版本历史"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {'insights': {}, 'history': []}
        return {'insights': {}, 'history': []}
    
    def _save_versions(self):
        """保存版本历史"""
        try:
            with open(self.versions_file, 'w', encoding='utf-8') as f:
                json.dump(self.versions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Integrator] ⚠️ 保存版本历史失败: {e}")
    
    def integrate(self, 
                  skill_name: str, 
                  validation_result: Dict[str, Any],
                  force: bool = False) -> Dict[str, Any]:
        """
        集成洞察到系统
        
        返回格式:
        {
            'integrated': bool,
            'version': str,
            'ab_test_required': bool,
            'rollback_available': bool,
            'message': str
        }
        """
        result = {
            'integrated': False,
            'version': None,
            'ab_test_required': False,
            'rollback_available': False,
            'message': ''
        }
        
        # 检查验证结果
        if not force and validation_result.get('recommendation') != 'INTEGRATE':
            result['message'] = f"验证评分不足: {validation_result.get('score', 0):.2f} < 0.8"
            return result
        
        try:
            # 1. 复制到active目录
            source_path = self.skills_dir / f"{skill_name}.py"
            if not source_path.exists():
                result['message'] = f"技能文件不存在: {source_path}"
                return result
            
            # 生成版本号
            version = f"v{int(time.time())}"
            active_path = self.active_dir / f"{skill_name}_{version}.py"
            
            shutil.copy2(source_path, active_path)
            
            # 2. 更新版本记录
            if skill_name not in self.versions['insights']:
                self.versions['insights'][skill_name] = {
                    'versions': [],
                    'current': None,
                    'previous': None
                }
            
            insight_versions = self.versions['insights'][skill_name]
            
            # 保存当前版本为previous
            if insight_versions['current']:
                insight_versions['previous'] = insight_versions['current']
                result['rollback_available'] = True
            
            # 设置新版本为当前
            insight_versions['current'] = {
                'version': version,
                'path': str(active_path),
                'timestamp': time.time(),
                'validation_score': validation_result.get('score', 0),
                'integrated_at': datetime.now().isoformat()
            }
            
            insight_versions['versions'].append(insight_versions['current'])
            
            # 3. 记录集成历史
            self.versions['history'].append({
                'skill_name': skill_name,
                'version': version,
                'action': 'INTEGRATE',
                'timestamp': time.time(),
                'validation': validation_result
            })
            
            self._save_versions()
            
            # 4. 热加载模块
            loaded = self._hot_load(skill_name, active_path)
            if loaded:
                self.active_insights[skill_name] = {
                    'version': version,
                    'module': loaded,
                    'path': str(active_path)
                }
            
            result['integrated'] = True
            result['version'] = version
            result['ab_test_required'] = True  # 需要A/B测试验证效果
            result['message'] = f"成功集成 {skill_name} {version}"
            
            print(f"   [Integrator] ✅ {result['message']}")
            
            return result
            
        except Exception as e:
            result['message'] = f"集成失败: {str(e)}"
            print(f"   [Integrator] ❌ {result['message']}")
            return result
    
    def _hot_load(self, skill_name: str, path: Path) -> Optional[Any]:
        """热加载Python模块"""
        try:
            spec = importlib.util.spec_from_file_location(skill_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[skill_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"   [Integrator] ⚠️ 热加载失败 {skill_name}: {e}")
            return None
    
    def run_ab_test(self, 
                    skill_name: str,
                    test_function,
                    iterations: int = 10) -> Dict[str, Any]:
        """
        A/B测试 - 对比集成前后的性能
        
        Args:
            skill_name: 技能名称
            test_function: 测试函数，应返回性能指标 (float)
            iterations: 测试迭代次数
        
        Returns:
            {
                'baseline': float,
                'with_insight': float,
                'improvement': float,
                'recommendation': str  # 'KEEP', 'ROLLBACK'
            }
        """
        if skill_name not in self.active_insights:
            return {'error': f'{skill_name} 未激活'}
        
        try:
            # 获取baseline（禁用洞察）
            self._disable_insight(skill_name)
            baseline_scores = [test_function() for _ in range(iterations)]
            baseline_avg = sum(baseline_scores) / len(baseline_scores)
            
            # 启用洞察后测试
            self._enable_insight(skill_name)
            insight_scores = [test_function() for _ in range(iterations)]
            insight_avg = sum(insight_scores) / len(insight_scores)
            
            # 计算改进
            improvement = ((insight_avg - baseline_avg) / baseline_avg * 100) if baseline_avg != 0 else 0
            
            result = {
                'baseline': baseline_avg,
                'with_insight': insight_avg,
                'improvement': improvement,
                'recommendation': 'KEEP' if improvement > 5 else 'ROLLBACK',  # >5%改进才保留
                'timestamp': time.time()
            }
            
            self.ab_test_results.append({
                'skill_name': skill_name,
                **result
            })
            
            # 记录到版本历史
            self.versions['history'].append({
                'skill_name': skill_name,
                'action': 'AB_TEST',
                'result': result,
                'timestamp': time.time()
            })
            self._save_versions()
            
            print(f"   [Integrator] 📊 A/B测试 {skill_name}: baseline={baseline_avg:.3f}, new={insight_avg:.3f}, improvement={improvement:+.1f}%")
            
            return result
            
        except Exception as e:
            return {'error': f'A/B测试失败: {str(e)}'}
    
    def _disable_insight(self, skill_name: str):
        """临时禁用洞察"""
        if skill_name in self.active_insights:
            self.active_insights[skill_name]['enabled'] = False
    
    def _enable_insight(self, skill_name: str):
        """启用洞察"""
        if skill_name in self.active_insights:
            self.active_insights[skill_name]['enabled'] = True
    
    def rollback(self, skill_name: str) -> Dict[str, Any]:
        """回滚到上一版本"""
        if skill_name not in self.versions['insights']:
            return {'success': False, 'message': f'{skill_name} 无版本记录'}
        
        insight_versions = self.versions['insights'][skill_name]
        previous = insight_versions.get('previous')
        
        if not previous:
            return {'success': False, 'message': '无可回滚版本'}
        
        try:
            # 恢复previous为current
            insight_versions['current'] = previous
            insight_versions['previous'] = None
            
            # 重新加载
            path = Path(previous['path'])
            if path.exists():
                loaded = self._hot_load(skill_name, path)
                if loaded:
                    self.active_insights[skill_name] = {
                        'version': previous['version'],
                        'module': loaded,
                        'path': str(path)
                    }
            
            # 记录回滚
            self.versions['history'].append({
                'skill_name': skill_name,
                'action': 'ROLLBACK',
                'to_version': previous['version'],
                'timestamp': time.time()
            })
            self._save_versions()
            
            print(f"   [Integrator] ⏪ 回滚 {skill_name} 到 {previous['version']}")
            
            return {'success': True, 'version': previous['version']}
            
        except Exception as e:
            return {'success': False, 'message': f'回滚失败: {str(e)}'}
    
    def archive_low_performers(self, threshold: float = 0.6) -> List[str]:
        """归档低效洞察"""
        archived = []
        
        for skill_name, insight_data in self.versions['insights'].items():
            current = insight_data.get('current')
            if not current:
                continue
            
            # 检查验证评分
            val_score = current.get('validation_score', 1.0)
            if val_score < threshold:
                try:
                    # 移动到archive目录
                    source = Path(current['path'])
                    if source.exists():
                        dest = self.archive_dir / source.name
                        shutil.move(str(source), str(dest))
                        archived.append(skill_name)
                        
                        # 更新版本记录
                        insight_data['current']['archived'] = True
                        insight_data['current']['archive_path'] = str(dest)
                        
                        print(f"   [Integrator] 📦 归档低效洞察: {skill_name} (score={val_score:.2f})")
                except Exception as e:
                    print(f"   [Integrator] ⚠️ 归档失败 {skill_name}: {e}")
        
        if archived:
            self._save_versions()
        
        return archived
    
    def get_active_insights(self) -> List[Dict[str, Any]]:
        """获取当前激活的所有洞察"""
        return [
            {
                'name': name,
                'version': data['version'],
                'path': data['path'],
                'enabled': data.get('enabled', True)
            }
            for name, data in self.active_insights.items()
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取集成统计信息"""
        total_insights = len(self.versions['insights'])
        active_count = len(self.active_insights)
        
        integrations = sum(1 for h in self.versions['history'] if h['action'] == 'INTEGRATE')
        rollbacks = sum(1 for h in self.versions['history'] if h['action'] == 'ROLLBACK')
        
        ab_tests = [r for r in self.ab_test_results]
        avg_improvement = sum(r['improvement'] for r in ab_tests) / len(ab_tests) if ab_tests else 0
        
        return {
            'total_insights': total_insights,
            'active_insights': active_count,
            'integrations': integrations,
            'rollbacks': rollbacks,
            'ab_tests_run': len(ab_tests),
            'average_improvement': avg_improvement,
            'keep_rate': sum(1 for r in ab_tests if r['recommendation'] == 'KEEP') / len(ab_tests) if ab_tests else 0
        }
