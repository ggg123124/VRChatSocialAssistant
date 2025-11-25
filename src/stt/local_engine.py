"""
本地 STT 引擎实现（基于 Faster-Whisper）
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .models import RecognitionResult, EngineConfig, ModelLoadError, RecognitionError

logger = logging.getLogger(__name__)


class LocalSTTEngine:
    """
    本地语音识别引擎
    
    使用 faster-whisper 进行本地语音识别
    """
    
    def __init__(self, config: EngineConfig):
        """
        初始化本地识别引擎
        
        Args:
            config: 引擎配置
        """
        self.config = config
        self.model = None
        self.device = None
        self.model_loaded = False
        
        # 统计信息
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        self.total_processing_time = 0.0
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化 Faster-Whisper 模型"""
        try:
            # 动态导入 faster-whisper
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ModelLoadError(
                    "faster-whisper 未安装。请运行: pip install faster-whisper"
                )
            
            # 确定设备
            if self.config.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            logger.info(f"使用设备: {self.device}")
            
            # 模型路径
            model_path = Path(self.config.model_name)
            
            # 如果是相对路径，则添加基础路径
            if not model_path.is_absolute():
                base_path = Path("models/stt/faster-whisper")
                model_path = base_path / self.config.model_name
            
            # 检查模型是否存在
            if model_path.exists():
                logger.info(f"从本地加载模型: {model_path}")
                model_name_or_path = str(model_path)
            else:
                # 使用预训练模型名称
                logger.info(f"使用预训练模型: {self.config.model_name}")
                model_name_or_path = self.config.model_name
            
            # 加载模型
            logger.info(f"正在加载模型: {model_name_or_path}, "
                       f"device={self.device}, compute_type={self.config.compute_type}")
            
            self.model = WhisperModel(
                model_name_or_path,
                device=self.device,
                compute_type=self.config.compute_type,
                download_root="models/stt/faster-whisper/"
            )
            
            self.model_loaded = True
            logger.info("模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            raise ModelLoadError(f"模型加载失败: {e}")
    
    def recognize(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        request_id: str = "",
        **kwargs
    ) -> RecognitionResult:
        """
        识别音频片段
        
        Args:
            audio_data: 音频数据（numpy array, float32）
            sample_rate: 采样率
            request_id: 请求ID
            **kwargs: 其他参数
        
        Returns:
            识别结果
        """
        if not self.model_loaded:
            return RecognitionResult(
                request_id=request_id,
                success=False,
                error_message="模型未加载"
            )
        
        start_time = time.time()
        self.total_recognitions += 1
        
        try:
            # 验证音频数据
            if not isinstance(audio_data, np.ndarray):
                raise RecognitionError("audio_data 必须是 numpy.ndarray")
            
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 确保音频是一维的
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # 计算音频时长
            duration = len(audio_data) / sample_rate
            
            # 检查音频时长
            if duration < 0.3:
                logger.warning(f"音频过短: {duration:.2f}s")
                return RecognitionResult(
                    request_id=request_id,
                    success=False,
                    duration=duration,
                    error_message="音频过短"
                )
            
            if duration > 30.0:
                logger.warning(f"音频过长: {duration:.2f}s，将被截断")
                audio_data = audio_data[:int(30.0 * sample_rate)]
                duration = 30.0
            
            # 获取语言参数
            language = kwargs.get('language', self.config.language)
            if language == "auto":
                language = None
            
            # 执行识别
            logger.debug(f"开始识别: duration={duration:.2f}s, language={language}")
            
            segments, info = self.model.transcribe(
                audio_data,
                language=language,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=self.config.temperature,
                vad_filter=self.config.vad_filter,
            )
            
            # 处理结果
            segments_list = list(segments)
            
            if not segments_list:
                logger.debug("未识别到文本")
                self.failed_recognitions += 1
                return RecognitionResult(
                    request_id=request_id,
                    success=False,
                    duration=duration,
                    processing_time=(time.time() - start_time) * 1000,
                    engine_type="local",
                    error_message="未识别到文本"
                )
            
            # 拼接所有分段文本
            full_text = " ".join([seg.text.strip() for seg in segments_list])
            
            # 计算平均置信度
            avg_confidence = sum([seg.avg_logprob for seg in segments_list]) / len(segments_list)
            # 转换为 0-1 范围（logprob 通常是负数）
            confidence = min(1.0, max(0.0, 1.0 + avg_confidence))
            
            # 构建分段结果
            from .models import SegmentResult
            segment_results = [
                SegmentResult(
                    id=i,
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    confidence=min(1.0, max(0.0, 1.0 + seg.avg_logprob))
                )
                for i, seg in enumerate(segments_list)
            ]
            
            # 记录处理时间
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            self.successful_recognitions += 1
            
            logger.info(f"识别成功: text='{full_text[:50]}...', "
                       f"confidence={confidence:.3f}, "
                       f"language={info.language}, "
                       f"time={processing_time:.1f}ms")
            
            return RecognitionResult(
                request_id=request_id,
                success=True,
                text=full_text,
                confidence=confidence,
                language=info.language,
                duration=duration,
                processing_time=processing_time,
                engine_type="local",
                segments=segment_results,
                speaker_id=kwargs.get('speaker_id'),
            )
            
        except Exception as e:
            self.failed_recognitions += 1
            logger.error(f"识别失败: {e}", exc_info=True)
            
            return RecognitionResult(
                request_id=request_id,
                success=False,
                duration=len(audio_data) / sample_rate if len(audio_data) > 0 else 0.0,
                processing_time=(time.time() - start_time) * 1000,
                engine_type="local",
                error_message=str(e)
            )
    
    def validate_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> tuple[bool, Optional[str]]:
        """
        验证音频数据是否符合要求
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
        
        Returns:
            (是否有效, 错误信息)
        """
        if not isinstance(audio_data, np.ndarray):
            return False, "audio_data 必须是 numpy.ndarray"
        
        if len(audio_data) == 0:
            return False, "音频数据为空"
        
        duration = len(audio_data) / sample_rate
        
        if duration < 0.3:
            return False, f"音频过短: {duration:.2f}s < 0.3s"
        
        if duration > 30.0:
            return False, f"音频过长: {duration:.2f}s > 30.0s"
        
        return True, None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计数据字典
        """
        avg_processing_time = (
            self.total_processing_time / self.total_recognitions
            if self.total_recognitions > 0 else 0.0
        )
        
        success_rate = (
            self.successful_recognitions / self.total_recognitions
            if self.total_recognitions > 0 else 0.0
        )
        
        return {
            'engine_type': 'local',
            'model_name': self.config.model_name,
            'device': self.device,
            'model_loaded': self.model_loaded,
            'total_recognitions': self.total_recognitions,
            'successful_recognitions': self.successful_recognitions,
            'failed_recognitions': self.failed_recognitions,
            'success_rate': success_rate,
            'avg_processing_time_ms': avg_processing_time,
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        self.total_processing_time = 0.0
        logger.info("统计信息已重置")
    
    def __del__(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None
