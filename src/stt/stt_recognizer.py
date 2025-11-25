"""
STT 识别器主接口类
"""

import logging
import uuid
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import yaml
import numpy as np

from .models import (
    RecognitionRequest,
    RecognitionResult,
    EngineConfig,
    EngineType,
    STTError
)
from .engine_manager import EngineManager, FallbackStrategy

logger = logging.getLogger(__name__)


class STTRecognizer:
    """
    STT 识别器主接口
    
    提供统一的语音识别接口，支持本地和云端引擎
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化识别器
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 引擎管理器
        self.engine_manager: Optional[EngineManager] = None
        
        # 全局回调函数
        self.callback: Optional[Callable[[RecognitionResult], None]] = None
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # 初始化
        self._initialize()
        
        logger.info("STTRecognizer 初始化完成")
    
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            配置字典
        """
        if config_path is None:
            config_path = "config/stt_config.yaml"
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return self._get_default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"加载配置文件: {config_path}")
            return config.get('stt', self._get_default_config())
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'default_engine': 'local',
            'fallback_enabled': True,
            'enable_cache': False,
            'max_audio_duration': 30.0,
            'min_audio_duration': 0.3,
            'local': {
                'model_path': 'models/stt/faster-whisper/',
                'model_size': 'medium',
                'device': 'auto',
                'compute_type': 'float16',
                'language': 'auto',
                'beam_size': 5,
                'temperature': 0.0,
                'vad_filter': False,
            },
            'cloud': {
                'provider': 'aliyun',
                'timeout': 5,
                'retry_count': 3,
            },
            'quality': {
                'min_confidence': 0.3,
                'max_text_length': 500,
            },
        }
    
    def _initialize(self):
        """初始化引擎管理器"""
        try:
            # 解析主引擎配置
            default_engine = self.config.get('default_engine', 'local')
            
            if default_engine == 'local':
                primary_config = self._parse_local_config()
                fallback_config = self._parse_cloud_config() if self.config.get('fallback_enabled') else None
            else:
                primary_config = self._parse_cloud_config()
                fallback_config = self._parse_local_config() if self.config.get('fallback_enabled') else None
            
            # 创建引擎管理器
            self.engine_manager = EngineManager(
                primary_config=primary_config,
                fallback_config=fallback_config,
                fallback_strategy=FallbackStrategy.AUTO,
                enable_fallback=self.config.get('fallback_enabled', True)
            )
            
            logger.info("引擎管理器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            raise STTError(f"初始化失败: {e}")
    
    def _parse_local_config(self) -> EngineConfig:
        """解析本地引擎配置"""
        local_config = self.config.get('local', {})
        
        model_size = local_config.get('model_size', 'medium')
        
        return EngineConfig(
            engine_type='local',
            model_name=model_size,
            device=local_config.get('device', 'auto'),
            language=local_config.get('language', 'auto'),
            compute_type=local_config.get('compute_type', 'float16'),
            beam_size=local_config.get('beam_size', 5),
            best_of=local_config.get('best_of', 5),
            temperature=local_config.get('temperature', 0.0),
            vad_filter=local_config.get('vad_filter', False),
        )
    
    def _parse_cloud_config(self) -> EngineConfig:
        """解析云端引擎配置"""
        cloud_config = self.config.get('cloud', {})
        provider = cloud_config.get('provider', 'aliyun')
        
        provider_config = cloud_config.get(provider, {})
        
        metadata = {}
        if provider == 'aliyun':
            metadata['access_key_id'] = provider_config.get('access_key_id')
        
        return EngineConfig(
            engine_type='cloud',
            cloud_provider=provider,
            api_key=provider_config.get('app_key') or provider_config.get('secret_id'),
            api_secret=provider_config.get('access_key_secret') or provider_config.get('secret_key'),
            timeout=cloud_config.get('timeout', 5),
            retry_count=cloud_config.get('retry_count', 3),
            metadata=metadata,
        )
    
    def recognize(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        **kwargs
    ) -> RecognitionResult:
        """
        同步识别音频
        
        Args:
            audio_data: 音频数据（numpy array, float32）
            sample_rate: 采样率
            **kwargs: 其他参数（language, speaker_id等）
        
        Returns:
            识别结果
        """
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        self.total_requests += 1
        
        try:
            # 验证音频数据
            valid, error_msg = self._validate_audio(audio_data, sample_rate)
            if not valid:
                self.failed_requests += 1
                return RecognitionResult(
                    request_id=request_id,
                    success=False,
                    error_message=error_msg
                )
            
            # 执行识别
            result = self.engine_manager.recognize(
                audio_data=audio_data,
                sample_rate=sample_rate,
                request_id=request_id,
                **kwargs
            )
            
            # 质量控制
            result = self._apply_quality_control(result)
            
            # 更新统计
            if result.success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            # 调用回调函数
            if self.callback:
                try:
                    self.callback(result)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {e}", exc_info=True)
            
            return result
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"识别异常: {e}", exc_info=True)
            
            return RecognitionResult(
                request_id=request_id,
                success=False,
                error_message=f"识别异常: {str(e)}"
            )
    
    def recognize_async(
        self,
        audio_data: np.ndarray,
        callback: Callable[[RecognitionResult], None],
        sample_rate: int = 16000,
        **kwargs
    ) -> str:
        """
        异步识别音频
        
        Args:
            audio_data: 音频数据
            callback: 回调函数
            sample_rate: 采样率
            **kwargs: 其他参数
        
        Returns:
            请求ID
        """
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 在新线程中执行识别
        import threading
        
        def async_recognize():
            result = self.recognize(
                audio_data=audio_data,
                sample_rate=sample_rate,
                **kwargs
            )
            result.request_id = request_id
            
            # 调用回调
            try:
                callback(result)
            except Exception as e:
                logger.error(f"异步回调执行失败: {e}", exc_info=True)
        
        thread = threading.Thread(target=async_recognize, daemon=True)
        thread.start()
        
        return request_id
    
    def _validate_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> tuple[bool, Optional[str]]:
        """
        验证音频数据
        
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
        
        min_duration = self.config.get('min_audio_duration', 0.3)
        max_duration = self.config.get('max_audio_duration', 30.0)
        
        if duration < min_duration:
            return False, f"音频过短: {duration:.2f}s < {min_duration}s"
        
        if duration > max_duration:
            return False, f"音频过长: {duration:.2f}s > {max_duration}s"
        
        return True, None
    
    def _apply_quality_control(self, result: RecognitionResult) -> RecognitionResult:
        """
        应用质量控制
        
        Args:
            result: 原始识别结果
        
        Returns:
            处理后的结果
        """
        if not result.success:
            return result
        
        quality_config = self.config.get('quality', {})
        
        # 置信度检查
        min_confidence = quality_config.get('min_confidence', 0.3)
        if result.confidence < min_confidence:
            logger.warning(f"置信度过低: {result.confidence:.3f} < {min_confidence}")
            result.metadata['low_confidence'] = True
        
        # 文本长度检查
        max_text_length = quality_config.get('max_text_length', 500)
        if len(result.text) > max_text_length:
            logger.warning(f"文本过长，截断: {len(result.text)} -> {max_text_length}")
            result.text = result.text[:max_text_length]
            result.metadata['truncated'] = True
        
        return result
    
    def set_callback(self, callback: Callable[[RecognitionResult], None]):
        """
        设置全局结果回调
        
        Args:
            callback: 回调函数
        """
        self.callback = callback
        logger.info("全局回调函数已设置")
    
    def switch_engine(self, engine_type: str) -> bool:
        """
        切换识别引擎
        
        Args:
            engine_type: 引擎类型（local/cloud）
        
        Returns:
            是否切换成功
        """
        if not self.engine_manager:
            logger.error("引擎管理器未初始化")
            return False
        
        return self.engine_manager.switch_engine(engine_type)
    
    def get_statistics(self) -> dict:
        """
        获取运行统计信息
        
        Returns:
            统计数据字典
        """
        engine_stats = self.engine_manager.get_statistics() if self.engine_manager else {}
        
        success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests > 0 else 0.0
        )
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'engine': engine_stats,
        }
    
    def reset_statistics(self):
        """重置统计计数器"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        if self.engine_manager:
            self.engine_manager.reset_statistics()
        
        logger.info("统计信息已重置")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        # 清理资源
        pass
