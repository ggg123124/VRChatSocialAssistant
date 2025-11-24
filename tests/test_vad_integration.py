"""
VAD 与音频采集模块集成测试

测试 VAD 模块与 AudioCapturer 的集成
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
import yaml

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from audio_capture.device_manager import DeviceManager
from audio_capture.audio_capturer import AudioCapturer
from vad.vad_detector import VADDetector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VADAudioIntegration:
    """VAD 与音频采集集成测试类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化集成测试
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'audio_config.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 音频采集器
        self.capturer = None
        
        # VAD 检测器
        self.vad_detector = None
        
        # 统计信息
        self.detected_segments = []
        self.total_audio_frames = 0
        
        logger.info("VAD 音频集成测试初始化完成")
    
    def setup_vad(self):
        """设置 VAD 检测器"""
        vad_config = self.config.get('vad', {})
        audio_config = self.config.get('audio', {})
        
        self.vad_detector = VADDetector(
            sample_rate=audio_config.get('samplerate', 16000),
            threshold=vad_config.get('threshold', 0.5),
            min_speech_duration_ms=vad_config.get('min_speech_duration_ms', 250),
            max_speech_duration_ms=vad_config.get('max_speech_duration_ms', 10000),
            min_silence_duration_ms=vad_config.get('min_silence_duration_ms', 300),
            speech_pad_ms=vad_config.get('speech_pad_ms', 30),
            window_size_samples=vad_config.get('window_size_samples', 512),
            device=vad_config.get('device', 'cpu'),
            debug=vad_config.get('debug', False)
        )
        
        # 设置语音片段回调
        def speech_callback(segment, metadata):
            self.detected_segments.append(metadata)
            logger.info(f"检测到语音片段 #{len(self.detected_segments)}: "
                       f"duration={metadata['duration']:.2f}s, "
                       f"confidence={metadata['avg_confidence']:.3f}, "
                       f"samples={metadata['num_samples']}")
        
        self.vad_detector.set_callback(speech_callback)
        
        logger.info("VAD 检测器已配置")
    
    def setup_audio_capturer(self, loopback_device=None, microphone_device=None):
        """
        设置音频采集器
        
        Args:
            loopback_device: WASAPI Loopback 设备索引
            microphone_device: 麦克风设备索引
        """
        audio_config = self.config.get('audio', {})
        
        self.capturer = AudioCapturer(
            loopback_device=loopback_device,
            microphone_device=microphone_device,
            samplerate=audio_config.get('samplerate', 16000),
            channels=audio_config.get('channels', 1),
            chunk_size=audio_config.get('chunk_size', 480)
        )
        
        # 设置回环音频回调（系统音频）
        def loopback_callback(audio_data, timestamp):
            self.total_audio_frames += 1
            # 将音频送入 VAD 检测器
            if self.vad_detector:
                self.vad_detector.process_audio(audio_data, timestamp)
        
        self.capturer.set_loopback_callback(loopback_callback)
        
        # 可选：设置麦克风回调
        def microphone_callback(audio_data, timestamp):
            # 这里可以单独处理麦克风音频
            # 目前我们主要关注系统音频（loopback）
            pass
        
        self.capturer.set_microphone_callback(microphone_callback)
        
        logger.info("音频采集器已配置")
    
    def run_test(self, duration_seconds: int = 10):
        """
        运行集成测试
        
        Args:
            duration_seconds: 测试时长（秒）
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"开始 VAD 与音频采集集成测试（时长: {duration_seconds} 秒）")
        logger.info(f"{'='*70}\n")
        
        # 启动音频采集
        logger.info("启动音频采集...")
        self.capturer.start()
        
        # 运行指定时长
        logger.info(f"正在采集音频并进行 VAD 检测...")
        logger.info("请播放一些音频（如音乐、视频）或对着麦克风说话\n")
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration_seconds:
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                
                # 每秒打印一次进度
                print(f"\r进度: {elapsed:.1f}s / {duration_seconds}s "
                      f"(剩余 {remaining:.1f}s) - "
                      f"音频帧: {self.total_audio_frames}, "
                      f"检测片段: {len(self.detected_segments)}", 
                      end='', flush=True)
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            logger.info("\n用户中断测试")
        
        finally:
            # 停止音频采集
            print("\n")
            logger.info("停止音频采集...")
            self.capturer.stop()
        
        # 打印结果
        self._print_results()
    
    def _print_results(self):
        """打印测试结果"""
        logger.info(f"\n{'='*70}")
        logger.info("测试结果")
        logger.info(f"{'='*70}\n")
        
        # 音频采集统计
        audio_stats = self.capturer.get_statistics()
        logger.info("音频采集统计:")
        logger.info(f"  Loopback 帧数: {audio_stats['loopback_frames_captured']}")
        logger.info(f"  麦克风帧数: {audio_stats['microphone_frames_captured']}")
        logger.info(f"  Loopback 溢出: {audio_stats['loopback_overflows']}")
        logger.info(f"  麦克风溢出: {audio_stats['microphone_overflows']}")
        
        # VAD 检测统计
        vad_stats = self.vad_detector.get_statistics()
        logger.info(f"\nVAD 检测统计:")
        logger.info(f"  处理帧数: {vad_stats['total_frames_processed']}")
        logger.info(f"  检测片段数: {vad_stats['speech_segments_detected']}")
        logger.info(f"  总语音时长: {vad_stats['total_speech_duration']:.2f}s")
        logger.info(f"  平均片段时长: {vad_stats['avg_speech_duration']:.2f}s")
        logger.info(f"  丢帧数: {vad_stats['frames_dropped']}")
        logger.info(f"  平均处理时间: {vad_stats['avg_processing_time_ms']:.2f}ms")
        logger.info(f"  当前状态: {vad_stats['current_state']}")
        logger.info(f"  缓冲区利用率: {vad_stats['buffer_utilization']:.1%}")
        
        # 性能评估
        logger.info(f"\n性能评估:")
        if vad_stats['avg_processing_time_ms'] < 30:
            logger.info(f"  ✓ 处理延迟: {vad_stats['avg_processing_time_ms']:.2f}ms (目标: <30ms) - 优秀")
        elif vad_stats['avg_processing_time_ms'] < 50:
            logger.info(f"  ⚠ 处理延迟: {vad_stats['avg_processing_time_ms']:.2f}ms (目标: <30ms) - 可接受")
        else:
            logger.info(f"  ✗ 处理延迟: {vad_stats['avg_processing_time_ms']:.2f}ms (目标: <30ms) - 需要优化")
        
        if vad_stats['frames_dropped'] == 0:
            logger.info(f"  ✓ 丢帧率: 0 - 完美")
        else:
            drop_rate = vad_stats['frames_dropped'] / max(vad_stats['total_frames_processed'], 1) * 100
            logger.info(f"  ⚠ 丢帧率: {drop_rate:.2f}% ({vad_stats['frames_dropped']} 帧)")
        
        # 语音片段详情
        if self.detected_segments:
            logger.info(f"\n语音片段详情:")
            for i, segment in enumerate(self.detected_segments[:10], 1):  # 只显示前 10 个
                logger.info(f"  片段 {i}: "
                           f"时长={segment['duration']:.2f}s, "
                           f"置信度={segment['avg_confidence']:.3f}, "
                           f"样本数={segment['num_samples']}")
            
            if len(self.detected_segments) > 10:
                logger.info(f"  ... (共 {len(self.detected_segments)} 个片段)")
        else:
            logger.info(f"\n⚠ 未检测到任何语音片段")
            logger.info(f"  可能原因:")
            logger.info(f"  1. 没有播放音频或说话")
            logger.info(f"  2. 音频设备未正确配置")
            logger.info(f"  3. VAD 阈值设置过高")
        
        logger.info(f"\n{'='*70}\n")


def main():
    """主函数"""
    print("=" * 70)
    print("VAD 与音频采集模块集成测试")
    print("=" * 70)
    
    # 显示可用设备
    print("\n正在检测音频设备...")
    device_manager = DeviceManager()
    
    loopback_devices = device_manager.list_loopback_devices()
    microphone_devices = device_manager.list_input_devices()
    
    print("\n可用的 WASAPI Loopback 设备（系统音频）:")
    for i, dev in enumerate(loopback_devices, 1):
        print(f"  {i}. [{dev['index']}] {dev['name']}")
    
    print("\n可用的麦克风设备:")
    for i, dev in enumerate(microphone_devices, 1):
        print(f"  {i}. [{dev['index']}] {dev['name']}")
    
    # 选择设备 - 使用默认设备
    loopback_device = None
    microphone_device = None
    
    # 重新打开设备管理器获取默认设备
    device_manager = DeviceManager()
    default_loopback = device_manager.get_default_wasapi_loopback()
    default_input = device_manager.get_default_input_device()
    
    if default_loopback:
        loopback_device = default_loopback['index']
        print(f"\n使用默认 Loopback 设备: [{loopback_device}] {default_loopback['name']}")
    elif loopback_devices:
        loopback_device = loopback_devices[0]['index']
        print(f"\n使用 Loopback 设备: [{loopback_device}] {loopback_devices[0]['name']}")
    
    if default_input:
        microphone_device = default_input['index']
        print(f"使用默认麦克风设备: [{microphone_device}] {default_input['name']}")
    elif microphone_devices:
        microphone_device = microphone_devices[0]['index']
        print(f"使用麦克风设备: [{microphone_device}] {microphone_devices[0]['name']}")
    
    device_manager.close()
    
    # 创建集成测试实例
    print("\n初始化 VAD 和音频采集模块...")
    integration = VADAudioIntegration()
    
    # 设置 VAD
    integration.setup_vad()
    
    # 设置音频采集器
    integration.setup_audio_capturer(
        loopback_device=loopback_device,
        microphone_device=microphone_device
    )
    
    # 运行测试
    test_duration = 10  # 默认 10 秒
    
    try:
        integration.run_test(duration_seconds=test_duration)
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
