"""
音频采集器

实现 WASAPI Loopback 音频和麦克风音频的同时采集
"""

import numpy as np
import pyaudiowpatch as pyaudio
from queue import Queue
from threading import Thread, Event
from typing import Optional, Callable, Tuple
import logging
import time
import struct
from scipy import signal

logger = logging.getLogger(__name__)


class AudioCapturer:
    """
    音频采集器
    
    同时采集 WASAPI Loopback 音频（扬声器输出）和麦克风输入
    """
    
    def __init__(
        self,
        loopback_device: Optional[int] = None,
        microphone_device: Optional[int] = None,
        samplerate: int = 16000,
        channels: int = 1,
        chunk_size: int = 480,  # 30ms @ 16kHz
        format: int = pyaudio.paInt16
    ):
        """
        初始化音频采集器
        
        Args:
            loopback_device: WASAPI Loopback 设备索引（用于采集系统音频）
            microphone_device: 麦克风设备索引
            samplerate: 采样率，默认 16kHz（STT 标准）
            channels: 声道数，默认单声道
            chunk_size: 缓冲区大小（样本数），默认 30ms
            format: 音频格式
        """
        self.loopback_device = loopback_device
        self.microphone_device = microphone_device
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        
        # PyAudio 实例
        self.p = pyaudio.PyAudio()
        
        # 音频队列
        self.loopback_queue = Queue()
        self.microphone_queue = Queue()
        
        # 流对象
        self.loopback_stream: Optional[pyaudio.Stream] = None
        self.microphone_stream: Optional[pyaudio.Stream] = None
        
        # 重采样相关
        self.loopback_native_rate = None  # 设备原生采样率
        self.resample_buffer = np.array([], dtype=np.float32)  # 重采样缓冲区
        
        # 控制标志
        self.is_running = False
        self._stop_event = Event()
        
        # 回调函数
        self.loopback_callback: Optional[Callable] = None
        self.microphone_callback: Optional[Callable] = None
        
        # 统计信息
        self.loopback_frames_captured = 0
        self.microphone_frames_captured = 0
        self.loopback_overflows = 0
        self.microphone_overflows = 0
        
        logger.info(f"AudioCapturer 初始化: sr={samplerate}, ch={channels}, chunk={chunk_size}")
    
    def set_loopback_callback(self, callback: Callable[[np.ndarray, float], None]):
        """
        设置回环音频回调函数
        
        Args:
            callback: 回调函数，接收参数 (audio_data: np.ndarray, timestamp: float)
        """
        self.loopback_callback = callback
    
    def set_microphone_callback(self, callback: Callable[[np.ndarray, float], None]):
        """
        设置麦克风音频回调函数
        
        Args:
            callback: 回调函数，接收参数 (audio_data: np.ndarray, timestamp: float)
        """
        self.microphone_callback = callback
    
    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        重采样音频数据
        
        Args:
            audio_data: 原始音频数据
            orig_sr: 原始采样率
            target_sr: 目标采样率
        
        Returns:
            重采样后的音频数据
        """
        if orig_sr == target_sr:
            return audio_data
        
        # 计算目标样本数
        num_samples = int(len(audio_data) * target_sr / orig_sr)
        
        # 使用 scipy 进行重采样
        resampled = signal.resample(audio_data, num_samples)
        
        return resampled.astype(np.float32)
    
    def _loopback_audio_callback(self, in_data, frame_count, time_info, status):
        """回环音频流回调"""
        if status:
            self.loopback_overflows += 1
            logger.warning(f"回环音频流状态: {status}")
        
        timestamp = time.time()
        
        # 将字节数据转换为 numpy 数组
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 如果是双声道，转换为单声道
        if self.channels == 1 and len(audio_data) == frame_count * 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        # 如果需要重采样（loopback设备采样率与目标采样率不同）
        if self.loopback_native_rate and self.loopback_native_rate != self.samplerate:
            audio_data = self._resample_audio(audio_data, self.loopback_native_rate, self.samplerate)
        
        # 放入队列
        self.loopback_queue.put((audio_data, timestamp))
        self.loopback_frames_captured += len(audio_data)
        
        # 调用用户回调
        if self.loopback_callback:
            try:
                self.loopback_callback(audio_data, timestamp)
            except Exception as e:
                logger.error(f"回环音频回调函数错误: {e}", exc_info=True)
        
        return (None, pyaudio.paContinue)
    
    def _microphone_audio_callback(self, in_data, frame_count, time_info, status):
        """麦克风音频流回调"""
        if status:
            self.microphone_overflows += 1
            logger.warning(f"麦克风音频流状态: {status}")
        
        timestamp = time.time()
        
        # 将字节数据转换为 numpy 数组
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 如果是双声道，转换为单声道
        if self.channels == 1 and len(audio_data) == frame_count * 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        # 放入队列
        self.microphone_queue.put((audio_data, timestamp))
        self.microphone_frames_captured += frame_count
        
        # 调用用户回调
        if self.microphone_callback:
            try:
                self.microphone_callback(audio_data, timestamp)
            except Exception as e:
                logger.error(f"麦克风音频回调函数错误: {e}", exc_info=True)
        
        return (None, pyaudio.paContinue)
    
    def start(self):
        """启动音频采集"""
        if self.is_running:
            logger.warning("音频采集已在运行")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # 启动 WASAPI Loopback 音频流
        if self.loopback_device is not None:
            try:
                # 获取设备信息
                device_info = self.p.get_device_info_by_index(self.loopback_device)
                
                # 使用设备的默认采样率和声道数
                loopback_channels = device_info.get('maxInputChannels', 2)
                self.loopback_native_rate = int(device_info['defaultSampleRate'])
                
                # 计算适合原生采样率的chunk大小
                native_chunk_size = int(self.chunk_size * self.loopback_native_rate / self.samplerate)
                
                self.loopback_stream = self.p.open(
                    format=self.format,
                    channels=loopback_channels,  # WASAPI loopback 通常是双声道
                    rate=self.loopback_native_rate,
                    input=True,
                    input_device_index=self.loopback_device,
                    frames_per_buffer=native_chunk_size,
                    stream_callback=self._loopback_audio_callback
                )
                self.loopback_stream.start_stream()
                logger.info(f"WASAPI Loopback 音频流已启动: device={self.loopback_device}, "
                           f"native_sr={self.loopback_native_rate}, target_sr={self.samplerate}, ch={loopback_channels}")
            except Exception as e:
                logger.error(f"启动 WASAPI Loopback 音频流失败: {e}", exc_info=True)
                self.loopback_stream = None
        else:
            logger.warning("未指定 WASAPI Loopback 设备，跳过系统音频采集")
        
        # 启动麦克风音频流
        if self.microphone_device is not None:
            try:
                device_info = self.p.get_device_info_by_index(self.microphone_device)
                mic_channels = min(self.channels, device_info.get('maxInputChannels', 1))
                
                self.microphone_stream = self.p.open(
                    format=self.format,
                    channels=mic_channels,
                    rate=self.samplerate,
                    input=True,
                    input_device_index=self.microphone_device,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._microphone_audio_callback
                )
                self.microphone_stream.start_stream()
                logger.info(f"麦克风音频流已启动: device={self.microphone_device}, "
                           f"sr={self.samplerate}, ch={mic_channels}")
            except Exception as e:
                logger.error(f"启动麦克风音频流失败: {e}", exc_info=True)
                self.microphone_stream = None
        else:
            logger.warning("未指定麦克风设备，跳过麦克风音频采集")
        
        logger.info("音频采集已启动")
    
    def stop(self):
        """停止音频采集"""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        # 停止 WASAPI Loopback 音频流
        if self.loopback_stream:
            try:
                self.loopback_stream.stop_stream()
                self.loopback_stream.close()
                logger.info("WASAPI Loopback 音频流已停止")
            except Exception as e:
                logger.error(f"停止 WASAPI Loopback 音频流错误: {e}")
            finally:
                self.loopback_stream = None
        
        # 停止麦克风音频流
        if self.microphone_stream:
            try:
                self.microphone_stream.stop_stream()
                self.microphone_stream.close()
                logger.info("麦克风音频流已停止")
            except Exception as e:
                logger.error(f"停止麦克风音频流错误: {e}")
            finally:
                self.microphone_stream = None
        
        logger.info("音频采集已停止")
    
    def get_loopback_audio(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """
        从队列获取回环音频数据（阻塞）
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            (音频数据, 时间戳) 或 None
        """
        try:
            return self.loopback_queue.get(timeout=timeout)
        except:
            return None
    
    def get_microphone_audio(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """
        从队列获取麦克风音频数据（阻塞）
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            (音频数据, 时间戳) 或 None
        """
        try:
            return self.microphone_queue.get(timeout=timeout)
        except:
            return None
    
    def get_statistics(self) -> dict:
        """
        获取采集统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'loopback_frames_captured': self.loopback_frames_captured,
            'microphone_frames_captured': self.microphone_frames_captured,
            'loopback_overflows': self.loopback_overflows,
            'microphone_overflows': self.microphone_overflows,
            'loopback_queue_size': self.loopback_queue.qsize(),
            'microphone_queue_size': self.microphone_queue.qsize(),
            'is_running': self.is_running
        }
    
    def clear_queues(self):
        """清空音频队列"""
        while not self.loopback_queue.empty():
            try:
                self.loopback_queue.get_nowait()
            except:
                break
        
        while not self.microphone_queue.empty():
            try:
                self.microphone_queue.get_nowait()
            except:
                break
        
        logger.info("音频队列已清空")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()
    
    def __del__(self):
        """析构函数"""
        self.stop()
        if hasattr(self, 'p'):
            self.p.terminate()
