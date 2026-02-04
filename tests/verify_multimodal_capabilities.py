#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Multimodal Capabilities Verification Script
AGI 多模态能力真实性验证脚本

功能:
1. 验证视觉感知 (摄像头)
2. 验证听觉感知 (麦克风 + Whisper ASR)
3. 验证表达能力 (扬声器 TTS)
4. 验证多模态融合逻辑 (Multimodal Fusion)
5. 生成评测报告数据

Author: AGI System
Date: 2025-12-03
"""

import sys
import os
import asyncio
import logging
import numpy as np
import json
from datetime import datetime

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_chat_enhanced import AGIChatInterface
from multimodal_fusion import MultimodalFusion, ModalityFeature

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultimodalTest")

async def verify_multimodal():
    print("\n" + "="*60)
    print("🧪 AGI Multimodal Capabilities Verification (真实性验证)")
    print("="*60 + "\n")
    
    results = {
        "vision": {"status": "pending", "details": ""},
        "hearing": {"status": "pending", "details": ""},
        "speech": {"status": "pending", "details": ""},
        "fusion": {"status": "pending", "details": ""}
    }

    # 1. 初始化 AGI Chat Interface
    print("🔄 Initializing AGI Interface (connecting to hardware)...")
    try:
        chat = AGIChatInterface()
        # 模拟初始化，因为我们只需要调用 handler
        # chat._initialize_system() # Constructor already calls this
        print("✅ AGI Interface Initialized")
    except Exception as e:
        print(f"❌ Failed to initialize AGI Interface: {e}")
        return

    # ---------------------------------------------------------
    # 2. 测试视觉 (Vision) - Capture Webcam
    # ---------------------------------------------------------
    print("\n👁️ Testing Vision (Webcam)...")
    try:
        # 尝试拍照
        vision_result = await chat._handle_capture_webcam(filename="test_vision_verify.jpg")
        
        if "error" in vision_result:
            print(f"⚠️ Vision Warning: {vision_result['error']}")
            results["vision"]["status"] = "failed_hardware"
            results["vision"]["details"] = vision_result['error']
        else:
            print(f"✅ Vision Success: Captured {vision_result['file_path']} ({vision_result.get('resolution')})")
            results["vision"]["status"] = "success"
            results["vision"]["details"] = f"Captured resolution: {vision_result.get('resolution')}"
            
    except Exception as e:
        print(f"❌ Vision Error: {e}")
        results["vision"]["status"] = "error"
        results["vision"]["details"] = str(e)

    # ---------------------------------------------------------
    # 3. 测试听觉 (Hearing) - Record Audio + ASR
    # ---------------------------------------------------------
    print("\n👂 Testing Hearing (Microphone + Whisper)...")
    try:
        # 录制 2 秒
        audio_result = await chat._handle_record_audio(duration=2, filename="test_audio_verify.wav")
        
        if "error" in audio_result:
            print(f"⚠️ Hearing Warning: {audio_result['error']}")
            results["hearing"]["status"] = "failed_hardware"
            results["hearing"]["details"] = audio_result['error']
        else:
            print(f"✅ Hearing Success: Recorded {audio_result['file_path']}")
            print(f"   Transcription: {audio_result.get('transcription')}")
            results["hearing"]["status"] = "success"
            results["hearing"]["details"] = f"ASR Result: {audio_result.get('transcription')}"

    except Exception as e:
        print(f"❌ Hearing Error: {e}")
        results["hearing"]["status"] = "error"
        results["hearing"]["details"] = str(e)

    # ---------------------------------------------------------
    # 4. 测试表达 (Speech) - TTS
    # ---------------------------------------------------------
    print("\n👄 Testing Speech (TTS)...")
    try:
        speech_result = await chat._handle_speak(text="Multimodal system verification in progress.")
        
        if "error" in speech_result:
            print(f"⚠️ Speech Warning: {speech_result['error']}")
            results["speech"]["status"] = "failed_hardware"
            results["speech"]["details"] = speech_result['error']
        else:
            print(f"✅ Speech Success: Audio output triggered.")
            results["speech"]["status"] = "success"
            results["speech"]["details"] = "TTS Triggered"

    except Exception as e:
        print(f"❌ Speech Error: {e}")
        results["speech"]["status"] = "error"
        results["speech"]["details"] = str(e)

    # ---------------------------------------------------------
    # 5. 测试融合 (Fusion) - Multimodal Logic
    # ---------------------------------------------------------
    print("\n🧠 Testing Multimodal Fusion Logic...")
    try:
        fusion = MultimodalFusion(unified_dim=128)
        
        # 模拟特征向量 (因为这里没有加载 CLIP 等重型模型)
        # 假设: 视觉特征 (2048维), 听觉特征 (1024维)
        vision_feat = ModalityFeature(
            modality="image",
            features=np.random.rand(2048).astype(np.float32),
            metadata={"source": "webcam"}
        )
        
        audio_feat = ModalityFeature(
            modality="audio",
            features=np.random.rand(1024).astype(np.float32),
            metadata={"source": "microphone"}
        )
        
        # 执行融合
        fused = fusion.fuse_modalities([vision_feat, audio_feat])
        
        print(f"✅ Fusion Success: Unified Vector Shape {fused.unified_features.shape}")
        print(f"   Contributions: {fused.modality_contributions}")
        
        # 测试跨模态关联 (Knowledge Graph Triple Generation)
        triples = fusion.generate_kg_triples([vision_feat, audio_feat])
        print(f"✅ Knowledge Generation: Created {len(triples)} triples")
        for t in triples:
            print(f"   - {t.subject} {t.predicate} {t.object} (conf={t.confidence:.2f})")

        results["fusion"]["status"] = "success"
        results["fusion"]["details"] = f"Fused {len(fused.modality_contributions)} modalities. Generated {len(triples)} triples."

    except Exception as e:
        print(f"❌ Fusion Error: {e}")
        results["fusion"]["status"] = "error"
        results["fusion"]["details"] = str(e)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("📊 Verification Summary")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    # Save results for report generation
    with open("multimodal_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(verify_multimodal())
