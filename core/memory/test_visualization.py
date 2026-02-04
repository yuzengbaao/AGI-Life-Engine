"""
测试数字神经大脑可视化功能
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

def test_visualization():
    """测试可视化器能否正常工作"""
    print("🧪 测试数字神经大脑可视化...")
    
    try:
        from core.memory.neural_memory import BiologicalMemorySystem
        
        # 初始化生物记忆系统
        print("   [1/4] 加载 BiologicalMemorySystem...")
        bio_mem = BiologicalMemorySystem()
        
        # 获取统计信息
        print("   [2/4] 获取拓扑统计...")
        stats = bio_mem.get_stats()
        print(f"         节点数: {stats['nodes']}")
        print(f"         记忆数: {stats['memories']}")
        print(f"         元数据: {stats['metadata_entries']}")
        
        # 生成可视化
        print("   [3/4] 生成可视化 HTML...")
        output_path = "./workspace/neural_brain_test.html"
        result = bio_mem.export_visualization(
            output_path=output_path,
            max_nodes=300,  # 限制节点数以加快渲染
        )
        
        print(f"         状态: {result.get('status')}")
        print(f"         渲染器: {result.get('renderer', 'unknown')}")
        print(f"         渲染节点: {result.get('nodes_rendered', 0)}")
        print(f"         渲染边: {result.get('edges_rendered', 0)}")
        
        # 验证文件
        print("   [4/4] 验证输出文件...")
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"         文件大小: {file_size:,} bytes")
            print(f"         ✅ 可视化成功: {output_path}")
            return True
        else:
            print("         ❌ 文件未生成")
            return False
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_visualization()
    sys.exit(0 if success else 1)
