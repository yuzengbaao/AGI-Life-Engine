#!/usr/bin/env python3
"""
🧹 意图历史清理脚本
[FIX 2026-01-17] 清理过去积压的意图，为系统提供干净的起点

用法:
    python scripts/cleanup_intent_history.py [--backup] [--dry-run]

选项:
    --backup    创建备份后再清理（默认）
    --dry-run   只显示将要清理的内容，不实际执行
    --force     不创建备份直接清理
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "intent_bridge"
USER_INTENTS_FILE = DATA_DIR / "user_intents.jsonl"
ENGINE_RESPONSES_FILE = DATA_DIR / "engine_responses.jsonl"
ACTIVE_INTENT_FILE = DATA_DIR / "active_intent.json"


def analyze_intents():
    """分析当前意图文件"""
    if not USER_INTENTS_FILE.exists():
        print("❌ 意图文件不存在")
        return None
    
    stats = {
        "total_lines": 0,
        "intents": 0,
        "confirmations": 0,
        "state_updates": 0,
        "pending": 0,
        "completed": 0,
        "failed": 0,
        "other": 0,
    }
    
    processed_ids = set()
    
    with open(USER_INTENTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total_lines"] += 1
            
            try:
                data = json.loads(line)
                
                if data.get("type") == "confirmation":
                    stats["confirmations"] += 1
                elif data.get("type") == "state_update":
                    stats["state_updates"] += 1
                    intent_id = data.get("intent_id")
                    new_state = data.get("new_state")
                    if intent_id and new_state in ['completed', 'failed', 'rejected']:
                        processed_ids.add(intent_id)
                elif 'id' in data and 'raw_input' in data:
                    stats["intents"] += 1
                    state = data.get('state', 'pending')
                    if state == 'pending':
                        stats["pending"] += 1
                    elif state == 'completed':
                        stats["completed"] += 1
                    elif state == 'failed':
                        stats["failed"] += 1
                    else:
                        stats["other"] += 1
                else:
                    stats["other"] += 1
            except json.JSONDecodeError:
                stats["other"] += 1
    
    stats["processed_by_state_update"] = len(processed_ids)
    return stats


def backup_files():
    """创建备份"""
    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backed_up = []
    for f in [USER_INTENTS_FILE, ENGINE_RESPONSES_FILE, ACTIVE_INTENT_FILE]:
        if f.exists():
            backup_path = backup_dir / f"{f.stem}_{timestamp}{f.suffix}"
            shutil.copy2(f, backup_path)
            backed_up.append(backup_path)
            print(f"   📦 备份: {f.name} -> {backup_path.name}")
    
    return backed_up


def cleanup_files():
    """清理文件"""
    for f in [USER_INTENTS_FILE, ENGINE_RESPONSES_FILE]:
        if f.exists():
            f.write_text("")
            print(f"   🧹 已清空: {f.name}")
    
    if ACTIVE_INTENT_FILE.exists():
        ACTIVE_INTENT_FILE.unlink()
        print(f"   🗑️  已删除: {ACTIVE_INTENT_FILE.name}")


def main():
    print("=" * 60)
    print("🧹 意图历史清理工具")
    print("=" * 60)
    
    # 解析参数
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    backup = "--backup" in sys.argv or not force
    
    # 分析当前状态
    print("\n📊 当前意图队列分析:")
    stats = analyze_intents()
    if stats:
        print(f"   总行数: {stats['total_lines']}")
        print(f"   意图数: {stats['intents']}")
        print(f"   确认消息: {stats['confirmations']}")
        print(f"   状态更新: {stats['state_updates']}")
        print(f"   待处理: {stats['pending']}")
        print(f"   已完成: {stats['completed']}")
        print(f"   已失败: {stats['failed']}")
        print(f"   已通过状态更新处理: {stats['processed_by_state_update']}")
    
    if dry_run:
        print("\n🔍 [DRY RUN] 不实际执行清理")
        print("   要实际执行，请移除 --dry-run 参数")
        return
    
    # 确认
    if not force:
        print("\n⚠️  即将清理所有历史意图！")
        confirm = input("   确认清理? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("   ❌ 已取消")
            return
    
    # 备份
    if backup:
        print("\n📦 创建备份...")
        backup_files()
    
    # 清理
    print("\n🧹 执行清理...")
    cleanup_files()
    
    print("\n" + "=" * 60)
    print("✅ 清理完成！")
    print("   意图队列已重置，系统将从干净状态开始")
    print("=" * 60)


if __name__ == "__main__":
    main()
