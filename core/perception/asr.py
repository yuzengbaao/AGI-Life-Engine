"""
Whisper ASR集成模块
支持中文语音识别和实时流式处理
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import queue
import threading
import time

logger = logging.getLogger(__name__)


class WhisperModelSize(Enum):
    """Whisper模型大小"""
    TINY = "tiny"           # 39M, 最快
    BASE = "base"           # 74M, 平衡
    SMALL = "small"         # 244M, 较好
    MEDIUM = "medium"       # 769M, 很好
    LARGE = "large"         # 1550M, 最好
    LARGE_V2 = "large-v2"   # 1550M, 最新
    LARGE_V3 = "large-v3"   # 1550M, 最新增强


class Language(Enum):
    """支持的语言"""
    AUTO = "auto"           # 自动检测
    CHINESE = "zh"          # 中文
    ENGLISH = "en"          # 英文
    JAPANESE = "ja"         # 日文
    KOREAN = "ko"           # 韩文
    SPANISH = "es"          # 西班牙文
    FRENCH = "fr"           # 法文
    GERMAN = "de"           # 德文


@dataclass
class WhisperResult:
    """Whisper识别结果"""
    text: str                           # 识别文本
    language: str                       # 检测语言
    confidence: float                   # 置信度
    segments: List[Dict[str, Any]]      # 分段信息
    processing_time: float              # 处理时间
    is_streaming: bool = False          # 是否流式


class WhisperASR:
    """
    Whisper ASR引擎
    支持离线语音识别和实时流式处理
    """
    
    def __init__(
        self,
        model_size: WhisperModelSize = WhisperModelSize.BASE,
        device: str = "auto",
        language: Language = Language.AUTO,
        use_faster_whisper: bool = True
    ):
        """
        初始化Whisper ASR
        
        Args:
            model_size: 模型大小
            device: 设备 (cuda/cpu/auto)
            language: 默认语言
            use_faster_whisper: 是否使用faster-whisper优化版本
        """
        self.model_size = model_size
        self.language = language
        self.use_faster_whisper = use_faster_whisper
        
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.is_loaded = False
        
        logger.info(f"🎤 初始化Whisper ASR引擎")
        logger.info(f"   - 模型: {model_size.value}")
        logger.info(f"   - 设备: {self.device}")
        logger.info(f"   - 语言: {language.value}")
        logger.info(f"   - 优化版本: {use_faster_whisper}")
    
    def load_model(self) -> bool:
        """
        加载Whisper模型
        
        Returns:
            是否加载成功
        """
        if self.is_loaded:
            return True
            
        try:
            logger.info("⏳ 正在加载Whisper模型...")
            
            # 查找本地模型路径
            import os
            
            # 1. 优先检查环境变量
            env_local_path = os.environ.get("WHISPER_MODEL_LOCAL_DIR")
            if env_local_path and os.path.exists(env_local_path):
                local_model_path = env_local_path
            else:
                # 2. 自动探测
                current_path = os.path.abspath(__file__)
                root_dir = os.path.dirname(current_path)
                for _ in range(3):
                    if os.path.exists(os.path.join(root_dir, "agi_chat_enhanced.py")):
                        break
                    root_dir = os.path.dirname(root_dir)
                
                # Check potential paths
                potential_paths = [
                    os.path.join(root_dir, "models", "faster-whisper-base"),
                    os.path.join(root_dir, "models", "whisper-base")
                ]
                
                local_model_path = os.path.join(root_dir, "models", "whisper-base") # default fallback
                for path in potential_paths:
                    if os.path.exists(path) and os.path.exists(os.path.join(path, "model.bin")):
                        local_model_path = path
                        break
            
            if self.use_faster_whisper:
                from faster_whisper import WhisperModel
                
                compute_type = "float16" if self.device == "cuda" else "int8"
                
                # 优先使用本地模型
                # check if model.bin exists (required for CTranslate2 format used by faster-whisper)
                is_valid_ct2_model = os.path.exists(os.path.join(local_model_path, "model.bin"))
                
                if os.path.exists(local_model_path) and is_valid_ct2_model:
                    logger.info(f"✅ 检测到本地Whisper模型(CTranslate2格式): {local_model_path}")
                    model_path_or_size = local_model_path
                else:
                    if os.path.exists(local_model_path) and not is_valid_ct2_model:
                        logger.warning(f"⚠️ 本地路径 {local_model_path} 存在但缺少 model.bin (非CTranslate2格式)，将忽略并自动下载/加载 {self.model_size.value}")
                    else:
                        logger.info(f"⚠️ 本地模型未找到，将自动下载: {self.model_size.value}")
                    model_path_or_size = self.model_size.value
                
                self.model = WhisperModel(
                    model_path_or_size,
                    device=self.device,
                    compute_type=compute_type
                )
            else:
                import whisper
                
                # OpenAI Whisper 库目前主要支持从 hub 加载或指定 .pt 文件
                # 这里简单处理，如果本地有文件则使用，否则下载
                self.model = whisper.load_model(self.model_size.value, device=self.device)
            
            self.is_loaded = True
            logger.info("✅ Whisper模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ Whisper模型加载失败: {e}")
            return False
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[Language] = None,
        task: str = "transcribe",
        **kwargs
    ) -> WhisperResult:
        """
        转录音频
        
        Args:
            audio: 音频文件路径或numpy数组
            language: 指定语言(None则使用默认)
            task: 任务类型 (transcribe转录 / translate翻译)
            **kwargs: 其他参数
            
        Returns:
            识别结果
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Whisper模型未加载")
        
        start_time = time.time()
        
        # 确定语言
        lang = language.value if language else self.language.value
        if lang == "auto":
            lang = None  # None表示自动检测
        
        try:
            if self.use_faster_whisper:
                # faster-whisper API
                segments_gen, info = self.model.transcribe(
                    audio,
                    language=lang,
                    task=task,
                    **kwargs
                )
                
                # 收集所有分段
                segments = []
                full_text = []
                
                for segment in segments_gen:
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "confidence": segment.avg_logprob
                    })
                    full_text.append(segment.text)
                
                result_text = " ".join(full_text).strip()
                detected_language = info.language
                confidence = info.language_probability
                
            else:
                # 标准whisper API
                result = self.model.transcribe(
                    audio,
                    language=lang,
                    task=task,
                    **kwargs
                )
                
                result_text = result["text"].strip()
                detected_language = result.get("language", lang or "unknown")
                
                # 计算平均置信度
                segments = result.get("segments", [])
                if segments:
                    confidences = [s.get("avg_logprob", 0) for s in segments]
                    confidence = np.exp(np.mean(confidences))
                else:
                    confidence = 0.5
            
            processing_time = time.time() - start_time
            
            return WhisperResult(
                text=result_text,
                language=detected_language,
                confidence=float(confidence),
                segments=segments,
                processing_time=processing_time,
                is_streaming=False
            )
            
        except Exception as e:
            logger.error(f"❌ Whisper转录失败: {e}")
            raise
    
    def transcribe_chinese(
        self,
        audio: Union[str, np.ndarray],
        **kwargs
    ) -> WhisperResult:
        """
        中文语音识别(便捷方法)
        
        Args:
            audio: 音频文件路径或numpy数组
            **kwargs: 其他参数
            
        Returns:
            识别结果
        """
        return self.transcribe(
            audio,
            language=Language.CHINESE,
            **kwargs
        )
    
    def detect_language(
        self,
        audio: Union[str, np.ndarray],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        检测音频语言
        
        Args:
            audio: 音频文件路径或numpy数组
            top_k: 返回前k个可能的语言
            
        Returns:
            语言概率字典
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Whisper模型未加载")
        
        try:
            if self.use_faster_whisper:
                _, info = self.model.transcribe(audio, language=None)
                # faster-whisper只返回最可能的语言
                return {info.language: info.language_probability}
            else:
                # 标准whisper支持语言检测
                import whisper
                
                # 加载音频
                if isinstance(audio, str):
                    audio_array = whisper.load_audio(audio)
                else:
                    audio_array = audio
                
                audio_array = whisper.pad_or_trim(audio_array)
                mel = whisper.log_mel_spectrogram(audio_array).to(self.device)
                
                # 检测语言
                _, probs = self.model.detect_language(mel)
                
                # 返回top_k
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_probs[:top_k])
                
        except Exception as e:
            logger.error(f"❌ 语言检测失败: {e}")
            return {}


class StreamingWhisperASR:
    """
    实时流式Whisper ASR
    支持音频块的连续识别
    """
    
    def __init__(
        self,
        whisper_asr: WhisperASR,
        chunk_duration: float = 3.0,
        overlap_duration: float = 0.5,
        sample_rate: int = 16000
    ):
        """
        初始化流式ASR
        
        Args:
            whisper_asr: Whisper ASR实例
            chunk_duration: 音频块时长(秒)
            overlap_duration: 重叠时长(秒)
            sample_rate: 采样率
        """
        self.asr = whisper_asr
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        
        # 音频缓冲区
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # 结果队列
        self.result_queue = queue.Queue()
        
        # 处理线程
        self.processing = False
        self.process_thread = None
        
        logger.info(f"🎤 初始化流式Whisper ASR")
        logger.info(f"   - 块时长: {chunk_duration}s")
        logger.info(f"   - 重叠: {overlap_duration}s")
        logger.info(f"   - 采样率: {sample_rate}Hz")
    
    def start(self):
        """启动流式处理"""
        if self.processing:
            logger.warning("⚠️ 流式处理已在运行")
            return
        
        self.processing = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info("✅ 流式处理已启动")
    
    def stop(self):
        """停止流式处理"""
        if not self.processing:
            return
        
        self.processing = False
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        logger.info("✅ 流式处理已停止")
    
    def add_audio(self, audio_chunk: np.ndarray):
        """
        添加音频块
        
        Args:
            audio_chunk: 音频数据(1D numpy array)
        """
        with self.buffer_lock:
            self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
    
    def _process_loop(self):
        """处理循环"""
        while self.processing:
            try:
                # 检查缓冲区是否有足够数据
                with self.buffer_lock:
                    if len(self.audio_buffer) < self.chunk_samples:
                        time.sleep(0.1)
                        continue
                    
                    # 提取音频块
                    audio_chunk = self.audio_buffer[:self.chunk_samples].copy()
                    
                    # 保留重叠部分
                    self.audio_buffer = self.audio_buffer[
                        self.chunk_samples - self.overlap_samples:
                    ]
                
                # --- 简单 VAD (语音活动检测) ---
                # 计算均方根 (RMS) 能量
                rms = np.sqrt(np.mean(audio_chunk**2))
                # 阈值需要根据麦克风调整，0.01 是一个保守值 (假设 float32 范围 -1.0 到 1.0)
                # 如果太安静，跳过识别，节省 CPU
                if rms < 0.005: 
                    # logger.debug(f"🤫 Silence detected (RMS: {rms:.4f}), skipping transcription.")
                    continue

                # 识别音频块
                start_time = time.time()
                
                result = self.asr.transcribe(audio_chunk)
                
                # 如果结果为空或置信度太低，也忽略 (Double Check)
                if not result.text.strip():
                    continue
                    
                result.is_streaming = True
                result.processing_time = time.time() - start_time
                
                # 放入结果队列
                self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"❌ 流式处理错误: {e}")
                time.sleep(0.1)
    
    def get_result(self, timeout: float = 0.1) -> Optional[WhisperResult]:
        """
        获取识别结果
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            识别结果或None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_buffer(self):
        """清空音频缓冲区"""
        with self.buffer_lock:
            self.audio_buffer = np.array([], dtype=np.float32)
        
        # 清空结果队列
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("✅ 缓冲区已清空")


# 便捷函数
def quick_transcribe(
    audio: Union[str, np.ndarray],
    language: str = "auto",
    model_size: str = "base"
) -> str:
    """
    快速转录(简化接口)
    
    Args:
        audio: 音频文件路径或numpy数组
        language: 语言代码 (auto/zh/en/ja等)
        model_size: 模型大小 (tiny/base/small/medium/large)
        
    Returns:
        识别文本
    """
    # 创建ASR实例
    asr = WhisperASR(
        model_size=WhisperModelSize(model_size),
        language=Language(language)
    )
    
    # 加载模型
    if not asr.load_model():
        raise RuntimeError("模型加载失败")
    
    # 转录
    result = asr.transcribe(audio)
    
    return result.text


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("🎤 Whisper ASR测试")
    print("="*70)
    
    try:
        # 测试1: 加载模型
        print("\n1️⃣ 测试模型加载...")
        asr = WhisperASR(
            model_size=WhisperModelSize.BASE,
            language=Language.CHINESE
        )
        
        if asr.load_model():
            print("✅ 模型加载成功")
        else:
            print("❌ 模型加载失败")
            exit(1)
        
        # 测试2: 生成测试音频
        print("\n2️⃣ 生成测试音频...")
        test_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3秒
        print("✅ 测试音频生成完成")
        
        # 测试3: 转录
        print("\n3️⃣ 测试转录...")
        result = asr.transcribe(test_audio)
        print(f"✅ 转录完成")
        print(f"   - 文本: {result.text}")
        print(f"   - 语言: {result.language}")
        print(f"   - 置信度: {result.confidence:.2f}")
        print(f"   - 处理时间: {result.processing_time:.4f}s")
        
        # 测试4: 流式ASR
        print("\n4️⃣ 测试流式ASR...")
        streaming_asr = StreamingWhisperASR(asr)
        streaming_asr.start()
        
        # 模拟添加音频块
        for i in range(3):
            chunk = np.random.randn(16000).astype(np.float32)  # 1秒
            streaming_asr.add_audio(chunk)
            time.sleep(0.5)
        
        # 获取结果
        time.sleep(2)
        result = streaming_asr.get_result(timeout=1.0)
        if result:
            print(f"✅ 流式识别完成")
            print(f"   - 文本: {result.text}")
        
        streaming_asr.stop()
        
        print("\n" + "="*70)
        print("✅ 所有测试完成!")
        print("="*70)
        
    except ImportError as e:
        print(f"\n❌ 缺少依赖: {e}")
        print("\n安装命令:")
        print("  pip install openai-whisper")
        print("  pip install faster-whisper  # 可选,但推荐")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
