"""
云端 STT 引擎实现（阿里云/腾讯云）
"""

import logging
import time
import json
import base64
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np

from .models import RecognitionResult, EngineConfig, RecognitionError

logger = logging.getLogger(__name__)


class CloudSTTEngineBase(ABC):
    """云端识别引擎基类"""
    
    def __init__(self, config: EngineConfig):
        """
        初始化云端引擎
        
        Args:
            config: 引擎配置
        """
        self.config = config
        
        # 统计信息
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        self.total_processing_time = 0.0
    
    @abstractmethod
    def recognize(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        request_id: str = "",
        **kwargs
    ) -> RecognitionResult:
        """识别音频片段"""
        pass
    
    def validate_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> tuple[bool, Optional[str]]:
        """验证音频数据"""
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
        """获取统计信息"""
        avg_processing_time = (
            self.total_processing_time / self.total_recognitions
            if self.total_recognitions > 0 else 0.0
        )
        
        success_rate = (
            self.successful_recognitions / self.total_recognitions
            if self.total_recognitions > 0 else 0.0
        )
        
        return {
            'engine_type': 'cloud',
            'provider': self.config.cloud_provider,
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


class AliyunSTTEngine(CloudSTTEngineBase):
    """
    阿里云语音识别引擎
    
    使用阿里云一句话识别API
    """
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.app_key = config.api_key
        self.access_key_id = config.metadata.get('access_key_id')
        self.access_key_secret = config.api_secret
        
        if not all([self.app_key, self.access_key_id, self.access_key_secret]):
            logger.warning("阿里云API凭证不完整，云端识别可能失败")
    
    def recognize(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        request_id: str = "",
        **kwargs
    ) -> RecognitionResult:
        """
        使用阿里云API识别音频
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            request_id: 请求ID
            **kwargs: 其他参数
        
        Returns:
            识别结果
        """
        start_time = time.time()
        self.total_recognitions += 1
        
        try:
            # 验证音频
            valid, error_msg = self.validate_audio(audio_data, sample_rate)
            if not valid:
                self.failed_recognitions += 1
                return RecognitionResult(
                    request_id=request_id,
                    success=False,
                    error_message=error_msg,
                    engine_type="cloud_aliyun"
                )
            
            # 转换音频为 PCM 格式
            audio_pcm = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_pcm.tobytes()
            
            # 这里应该调用阿里云API
            # 由于需要实际的API凭证，这里提供模拟实现
            logger.warning("阿里云API未实际调用（需要配置凭证）")
            
            # 模拟API调用
            duration = len(audio_data) / sample_rate
            
            # 模拟返回结果
            result_text = "[阿里云识别-模拟]"
            confidence = 0.85
            
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            self.successful_recognitions += 1
            
            logger.info(f"阿里云识别成功（模拟）: time={processing_time:.1f}ms")
            
            return RecognitionResult(
                request_id=request_id,
                success=True,
                text=result_text,
                confidence=confidence,
                language="zh",
                duration=duration,
                processing_time=processing_time,
                engine_type="cloud_aliyun",
                speaker_id=kwargs.get('speaker_id'),
            )
            
        except Exception as e:
            self.failed_recognitions += 1
            logger.error(f"阿里云识别失败: {e}", exc_info=True)
            
            return RecognitionResult(
                request_id=request_id,
                success=False,
                duration=len(audio_data) / sample_rate if len(audio_data) > 0 else 0.0,
                processing_time=(time.time() - start_time) * 1000,
                engine_type="cloud_aliyun",
                error_message=str(e)
            )


class TencentSTTEngine(CloudSTTEngineBase):
    """
    腾讯云语音识别引擎
    
    使用腾讯云一句话识别API
    """
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.secret_id = config.api_key
        self.secret_key = config.api_secret
        
        if not all([self.secret_id, self.secret_key]):
            logger.warning("腾讯云API凭证不完整，云端识别可能失败")
    
    def recognize(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        request_id: str = "",
        **kwargs
    ) -> RecognitionResult:
        """
        使用腾讯云API识别音频
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            request_id: 请求ID
            **kwargs: 其他参数
        
        Returns:
            识别结果
        """
        start_time = time.time()
        self.total_recognitions += 1
        
        try:
            # 验证音频
            valid, error_msg = self.validate_audio(audio_data, sample_rate)
            if not valid:
                self.failed_recognitions += 1
                return RecognitionResult(
                    request_id=request_id,
                    success=False,
                    error_message=error_msg,
                    engine_type="cloud_tencent"
                )
            
            # 转换音频为 PCM 格式
            audio_pcm = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_pcm.tobytes()
            
            # 这里应该调用腾讯云API
            # 由于需要实际的API凭证，这里提供模拟实现
            logger.warning("腾讯云API未实际调用（需要配置凭证）")
            
            # 模拟API调用
            duration = len(audio_data) / sample_rate
            
            # 模拟返回结果
            result_text = "[腾讯云识别-模拟]"
            confidence = 0.87
            
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            self.successful_recognitions += 1
            
            logger.info(f"腾讯云识别成功（模拟）: time={processing_time:.1f}ms")
            
            return RecognitionResult(
                request_id=request_id,
                success=True,
                text=result_text,
                confidence=confidence,
                language="zh",
                duration=duration,
                processing_time=processing_time,
                engine_type="cloud_tencent",
                speaker_id=kwargs.get('speaker_id'),
            )
            
        except Exception as e:
            self.failed_recognitions += 1
            logger.error(f"腾讯云识别失败: {e}", exc_info=True)
            
            return RecognitionResult(
                request_id=request_id,
                success=False,
                duration=len(audio_data) / sample_rate if len(audio_data) > 0 else 0.0,
                processing_time=(time.time() - start_time) * 1000,
                engine_type="cloud_tencent",
                error_message=str(e)
            )


class CloudSTTEngine:
    """
    云端识别引擎工厂类
    """
    
    @staticmethod
    def create(config: EngineConfig) -> CloudSTTEngineBase:
        """
        创建云端引擎实例
        
        Args:
            config: 引擎配置
        
        Returns:
            云端引擎实例
        """
        provider = config.cloud_provider
        
        if provider == "aliyun":
            return AliyunSTTEngine(config)
        elif provider == "tencent":
            return TencentSTTEngine(config)
        else:
            raise ValueError(f"不支持的云服务提供商: {provider}")
