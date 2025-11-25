"""
VRChat 社交辅助工具 - 功能集成测试程序

提供命令行交互式界面，集成测试所有已开发的模块：
- 音频采集
- VAD 语音活动检测
- 说话人识别
- 语音转文本 (STT)
- 记忆管理

使用方法:
    python tests/integrated_test.py [选项]

选项:
    --help, -h          显示帮助信息
    --init              仅运行初始化检查
    --module <名称>     直接进入指定模块测试
    --full              直接运行完整流程测试
    --debug             启用调试模式
"""

import sys
import os
import logging
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# 导入测试工具
from tests.test_utils import (
    print_title, print_subtitle, print_separator,
    print_success, print_error, print_warning, print_info,
    show_menu, get_user_input, get_number_input, confirm,
    wait_for_enter, clear_screen, print_table,
    generate_test_audio, generate_speech_audio, generate_silence,
    PerformanceTimer, StatisticsCollector, format_timestamp,
    format_duration, safe_execute
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/integrated_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntegratedTest:
    """集成测试主类"""
    
    def __init__(self):
        """初始化测试环境"""
        self.modules_initialized = {
            'audio': False,
            'vad': False,
            'speaker': False,
            'stt': False,
            'memory': False
        }
        
        # 模块实例
        self.audio_capturer = None
        self.vad_detector = None
        self.speaker_recognizer = None
        self.stt_recognizer = None
        self.memory_manager = None
        
        # 统计收集器
        self.stats = StatisticsCollector()
        
        # 配置路径
        self.config_dir = project_root / 'config'
        self.data_dir = project_root / 'data'
        
        # 确保数据目录存在
        self.data_dir.mkdir(exist_ok=True)
    
    def run(self):
        """运行主程序"""
        print_title("VRChat 社交辅助工具 - 功能集成测试")
        print()
        print_info("欢迎使用集成测试工具！")
        print_info("本工具将帮助您测试所有已开发的模块功能。")
        print()
        
        # 显示主菜单
        self.show_main_menu()
    
    def show_main_menu(self):
        """显示主菜单"""
        while True:
            options = {
                '1': '系统初始化 - 检查环境并初始化所有模块',
                '2': '单模块测试 - 测试各个模块的独立功能',
                '3': '完整流程测试 - 运行端到端的语音处理流程',
                '4': '数据管理 - 管理好友档案和对话记录',
                '5': '系统信息 - 查看系统状态和统计信息'
            }
            
            choice = show_menu("主菜单", options)
            
            if choice == '0':
                if confirm("确定要退出吗？"):
                    self.cleanup()
                    print_success("已退出，再见！")
                    break
            elif choice == '1':
                self.initialize_system()
            elif choice == '2':
                self.show_module_test_menu()
            elif choice == '3':
                self.run_full_pipeline_test()
            elif choice == '4':
                self.show_data_management_menu()
            elif choice == '5':
                self.show_system_info()
            
            if choice != '0':
                wait_for_enter()
    
    def initialize_system(self):
        """初始化系统"""
        print_title("系统初始化")
        print()
        
        print_info("开始初始化检查...")
        print()
        
        # 1. 检查配置文件
        print_subtitle("1. 检查配置文件", "-")
        self.check_config_files()
        print()
        
        # 2. 检查数据目录
        print_subtitle("2. 检查数据目录", "-")
        self.check_data_directories()
        print()
        
        # 3. 初始化各模块
        print_subtitle("3. 初始化模块", "-")
        self.initialize_modules()
        print()
        
        # 显示初始化结果
        self.show_initialization_result()
    
    def check_config_files(self):
        """检查配置文件"""
        config_files = [
            'audio_config.yaml',
            'memory_config.yaml',
            'speaker_recognition_config.yaml',
            'stt_config.yaml'
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                print_success(f"{config_file} - 存在")
            else:
                print_warning(f"{config_file} - 不存在（将使用默认配置）")
    
    def check_data_directories(self):
        """检查数据目录"""
        directories = [
            'data',
            'data/profiles',
            'data/conversations',
            'data/vector_db',
            'data/speaker_profiles'
        ]
        
        for dir_path in directories:
            full_path = project_root / dir_path
            if full_path.exists():
                print_success(f"{dir_path}/ - 存在")
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                print_info(f"{dir_path}/ - 已创建")
    
    def initialize_modules(self):
        """初始化各模块"""
        # 音频采集模块
        print("初始化音频采集模块...")
        result = self.init_audio_module()
        if result:
            print_success("音频采集模块初始化成功")
        else:
            print_error("音频采集模块初始化失败")
        
        # VAD 模块
        print("\n初始化 VAD 模块...")
        result = self.init_vad_module()
        if result:
            print_success("VAD 模块初始化成功")
        else:
            print_error("VAD 模块初始化失败")
        
        # 说话人识别模块
        print("\n初始化说话人识别模块...")
        result = self.init_speaker_module()
        if result:
            print_success("说话人识别模块初始化成功")
        else:
            print_error("说话人识别模块初始化失败")
        
        # STT 模块
        print("\n初始化 STT 模块...")
        result = self.init_stt_module()
        if result:
            print_success("STT 模块初始化成功")
        else:
            print_error("STT 模块初始化失败")
        
        # 记忆管理模块
        print("\n初始化记忆管理模块...")
        result = self.init_memory_module()
        if result:
            print_success("记忆管理模块初始化成功")
        else:
            print_error("记忆管理模块初始化失败")
    
    def init_audio_module(self) -> bool:
        """初始化音频采集模块"""
        try:
            from audio_capture import DeviceManager
            device_manager = DeviceManager()
            devices = device_manager.list_devices()
            if devices:
                self.modules_initialized['audio'] = True
                return True
        except Exception as e:
            logger.error(f"音频模块初始化失败: {e}", exc_info=True)
        return False
    
    def init_vad_module(self) -> bool:
        """初始化 VAD 模块"""
        try:
            from vad import VADDetector
            self.vad_detector = VADDetector(
                sample_rate=16000,
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_ms=10000,
                min_silence_duration_ms=300
            )
            self.modules_initialized['vad'] = True
            return True
        except Exception as e:
            logger.error(f"VAD 模块初始化失败: {e}", exc_info=True)
        return False
    
    def init_speaker_module(self) -> bool:
        """初始化说话人识别模块"""
        try:
            from speaker_recognition import SpeakerRecognizer
            self.speaker_recognizer = SpeakerRecognizer()
            self.modules_initialized['speaker'] = True
            return True
        except Exception as e:
            logger.error(f"说话人识别模块初始化失败: {e}", exc_info=True)
        return False
    
    def init_stt_module(self) -> bool:
        """初始化 STT 模块"""
        try:
            from stt import STTRecognizer
            self.stt_recognizer = STTRecognizer()
            self.modules_initialized['stt'] = True
            return True
        except Exception as e:
            logger.error(f"STT 模块初始化失败: {e}", exc_info=True)
        return False
    
    def init_memory_module(self) -> bool:
        """初始化记忆管理模块"""
        try:
            from memory import MemoryManager
            config_path = self.config_dir / 'memory_config.yaml'
            if config_path.exists():
                self.memory_manager = MemoryManager(str(config_path))
            else:
                self.memory_manager = MemoryManager()
            self.modules_initialized['memory'] = True
            return True
        except Exception as e:
            logger.error(f"记忆管理模块初始化失败: {e}", exc_info=True)
        return False
    
    def show_initialization_result(self):
        """显示初始化结果"""
        print_subtitle("初始化结果", "=")
        
        total = len(self.modules_initialized)
        success = sum(self.modules_initialized.values())
        
        for module, status in self.modules_initialized.items():
            status_text = "✓ 成功" if status else "✗ 失败"
            if status:
                print_success(f"{module.upper()} 模块: {status_text}")
            else:
                print_error(f"{module.upper()} 模块: {status_text}")
        
        print()
        print(f"总计: {success}/{total} 模块初始化成功 ({success/total*100:.0f}%)")
        
        if success == total:
            print_success("\n✓ 所有模块初始化成功！")
        elif success > 0:
            print_warning(f"\n⚠ 部分模块初始化失败，部分功能可能不可用")
        else:
            print_error(f"\n✗ 所有模块初始化失败，请检查环境配置")
    
    def show_module_test_menu(self):
        """显示模块测试菜单"""
        while True:
            options = {
                '1': '音频采集测试',
                '2': 'VAD 检测测试',
                '3': '说话人识别测试',
                '4': 'STT 语音转文本测试',
                '5': '记忆管理测试'
            }
            
            choice = show_menu("单模块测试", options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.test_audio_capture()
            elif choice == '2':
                self.test_vad()
            elif choice == '3':
                self.test_speaker_recognition()
            elif choice == '4':
                self.test_stt()
            elif choice == '5':
                self.test_memory()
            
            if choice != '0':
                wait_for_enter()
    
    def test_audio_capture(self):
        """测试音频采集模块"""
        print_title("音频采集模块测试")
        
        if not self.modules_initialized['audio']:
            print_error("音频模块未初始化，请先运行系统初始化")
            return
        
        options = {
            '1': '设备枚举测试 - 列出所有音频设备',
            '2': '短时采集测试 - 采集5秒音频',
        }
        
        choice = show_menu("音频采集测试", options)
        
        if choice == '1':
            self.test_audio_devices()
        elif choice == '2':
            self.test_audio_capture_short()
    
    def test_audio_devices(self):
        """测试设备枚举"""
        print_subtitle("音频设备列表")
        
        try:
            from audio_capture import DeviceManager
            device_manager = DeviceManager()
            
            devices = device_manager.list_devices()
            
            if not devices:
                print_warning("未找到可用的音频设备")
                return
            
            print(f"\n找到 {len(devices)} 个音频设备:\n")
            
            headers = ["索引", "名称", "通道数", "采样率", "驱动"]
            rows = []
            
            for device in devices:
                rows.append([
                    str(device['index']),
                    device['name'][:40],
                    str(device.get('maxInputChannels', 0)),
                    str(device.get('defaultSampleRate', 0)),
                    device.get('hostApi', 'Unknown')
                ])
            
            print_table(headers, rows)
            
            # 显示默认设备
            print("\n默认设备:")
            loopback = device_manager.get_default_wasapi_loopback()
            if loopback:
                print_success(f"WASAPI Loopback: {loopback['name']}")
            else:
                print_warning("未找到 WASAPI Loopback 设备")
        
        except Exception as e:
            logger.error(f"设备枚举失败: {e}", exc_info=True)
            print_error(f"设备枚举失败: {e}")
    
    def test_audio_capture_short(self):
        """短时采集测试"""
        print_subtitle("短时采集测试（5秒）")
        
        try:
            from audio_capture import AudioCapturer, DeviceManager
            
            device_manager = DeviceManager()
            loopback = device_manager.get_default_wasapi_loopback()
            
            if not loopback:
                print_error("未找到 WASAPI Loopback 设备")
                return
            
            print_info("使用设备: " + loopback['name'])
            print_info("开始采集 5 秒音频...")
            print()
            
            capturer = AudioCapturer(
                loopback_device=loopback['index'],
                samplerate=16000,
                channels=1,
                chunk_size=480
            )
            
            frames = []
            
            def callback(audio_data, timestamp):
                frames.append(audio_data)
                if len(frames) % 10 == 0:
                    print(f"已采集: {len(frames)} 帧", end='\r')
            
            capturer.set_loopback_callback(callback)
            capturer.start()
            
            time.sleep(5)
            
            capturer.stop()
            
            print()
            print_success(f"采集完成！共采集 {len(frames)} 帧音频")
            
            if frames:
                audio_data = np.concatenate(frames)
                rms = np.sqrt(np.mean(audio_data ** 2))
                print(f"音频 RMS: {rms:.6f}")
                print(f"音频时长: {len(audio_data) / 16000:.2f} 秒")
        
        except Exception as e:
            logger.error(f"音频采集失败: {e}", exc_info=True)
            print_error(f"音频采集失败: {e}")
    
    def test_vad(self):
        """测试 VAD 模块"""
        print_title("VAD 语音活动检测测试")
        
        if not self.modules_initialized['vad']:
            print_error("VAD 模块未初始化，请先运行系统初始化")
            return
        
        print_info("使用合成音频进行 VAD 测试")
        print()
        
        try:
            # 生成测试音频序列：静音-语音-静音-语音-静音
            print_subtitle("生成测试音频")
            
            sample_rate = 16000
            frame_size = 480  # 30ms
            frames = []
            
            # 1. 静音 500ms
            print("生成静音片段 (500ms)...")
            silence1 = generate_silence(0.5, sample_rate)
            for i in range(0, len(silence1), frame_size):
                frames.append(silence1[i:i+frame_size])
            
            # 2. 语音 1 秒
            print("生成语音片段 (1000ms)...")
            speech1 = generate_speech_audio(1.0, sample_rate)
            for i in range(0, len(speech1), frame_size):
                frames.append(speech1[i:i+frame_size])
            
            # 3. 静音 500ms
            print("生成静音片段 (500ms)...")
            silence2 = generate_silence(0.5, sample_rate)
            for i in range(0, len(silence2), frame_size):
                frames.append(silence2[i:i+frame_size])
            
            # 4. 语音 800ms
            print("生成语音片段 (800ms)...")
            speech2 = generate_speech_audio(0.8, sample_rate)
            for i in range(0, len(speech2), frame_size):
                frames.append(speech2[i:i+frame_size])
            
            # 5. 静音 500ms
            print("生成静音片段 (500ms)...")
            silence3 = generate_silence(0.5, sample_rate)
            for i in range(0, len(silence3), frame_size):
                frames.append(silence3[i:i+frame_size])
            
            print_success(f"测试音频生成完成，共 {len(frames)} 帧")
            print()
            
            # 处理音频
            print_subtitle("VAD 检测")
            print("预期结果: 检测到 2 个语音片段")
            print()
            
            detected_segments = []
            
            def speech_callback(segment, metadata):
                detected_segments.append(metadata)
                print(f"\n✓ 检测到语音片段 #{len(detected_segments)}:")
                print(f"  时长: {metadata['duration']:.2f} 秒")
                print(f"  置信度: {metadata['avg_confidence']:.3f}")
                print(f"  样本数: {metadata['num_samples']}")
            
            self.vad_detector.set_callback(speech_callback)
            
            timestamp = time.time()
            for i, frame in enumerate(frames):
                if len(frame) == frame_size:
                    self.vad_detector.process_audio(frame, timestamp)
                    timestamp += 0.03
                
                if (i + 1) % 10 == 0:
                    print(f"处理进度: {i+1}/{len(frames)} 帧", end='\r')
            
            print()
            print()
            
            # 显示统计
            stats = self.vad_detector.get_statistics()
            print_subtitle("VAD 统计信息")
            print(f"处理帧数: {stats['total_frames_processed']}")
            print(f"检测片段数: {stats['speech_segments_detected']}")
            print(f"总语音时长: {stats['total_speech_duration']:.2f} 秒")
            print(f"平均处理时间: {stats['avg_processing_time_ms']:.2f} ms")
            print(f"丢帧数: {stats['frames_dropped']}")
            
            if stats['speech_segments_detected'] >= 1:
                print()
                print_success(f"测试成功！检测到 {len(detected_segments)} 个语音片段")
            else:
                print()
                print_warning("未检测到语音片段")
        
        except Exception as e:
            logger.error(f"VAD 测试失败: {e}", exc_info=True)
            print_error(f"VAD 测试失败: {e}")
    
    def test_speaker_recognition(self):
        """测试说话人识别模块"""
        print_title("说话人识别测试")
        
        if not self.modules_initialized['speaker']:
            print_error("说话人识别模块未初始化，请先运行系统初始化")
            return
        
        options = {
            '1': '注册新好友',
            '2': '识别测试',
            '3': '查看已注册好友'
        }
        
        choice = show_menu("说话人识别测试", options)
        
        if choice == '1':
            self.test_speaker_register()
        elif choice == '2':
            self.test_speaker_recognize()
        elif choice == '3':
            self.test_speaker_list()
    
    def test_speaker_register(self):
        """测试声纹注册"""
        print_subtitle("注册新好友")
        
        friend_name = get_user_input("请输入好友姓名")
        if not friend_name:
            print_warning("已取消")
            return
        
        # 生成好友ID（使用拼音或简化名称）
        # 对于中文名字，使用哈希值
        if friend_name == "也许一切都是不能" or friend_name == "不能":
            friend_id = "friend_buneng"
        elif friend_name == "尾翼稳定脱壳穿甲鱼" or friend_name == "阿鱼":
            friend_id = "friend_ayu"
        else:
            friend_id = f"friend_{int(time.time())}"
        
        print()
        print_info("选择注册方式：")
        print("  1. 使用合成音频（快速测试）")
        print("  2. 录制真实语音（推荐，用于实际使用）")
        print()
        
        choice = get_user_input("请选择 (1/2)", "2")
        
        if choice == "1":
            self._register_with_synthetic_audio(friend_id, friend_name)
        elif choice == "2":
            self._register_with_real_recording(friend_id, friend_name)
        else:
            print_warning("无效选择，已取消")
    
    def _register_with_synthetic_audio(self, friend_id: str, friend_name: str):
        """使用合成音频注册声纹"""
        print_info(f"正在为 {friend_name} 生成声纹样本（使用合成音频）...")
        
        # 生成3段音频样本
        audio_segments = []
        seed = hash(friend_name) % 1000
        
        for i in range(3):
            audio = generate_test_audio(duration=2.5, seed=seed + i)
            audio_segments.append(audio)
            print(f"  样本 {i+1}: {len(audio)/16000:.2f}秒")
        
        # 注册
        try:
            success = self.speaker_recognizer.register_speaker(
                friend_id=friend_id,
                name=friend_name,
                audio_segments=audio_segments,
                sample_rate=16000
            )
            
            if success:
                print_success(f"\n{friend_name} 注册成功！")
                print(f"好友ID: {friend_id}")
                print_warning("注意：合成音频仅用于测试，实际使用请录制真实语音")
            else:
                print_error(f"\n{friend_name} 注册失败")
        
        except Exception as e:
            logger.error(f"声纹注册失败: {e}", exc_info=True)
            print_error(f"注册失败: {e}")
    
    def _register_with_real_recording(self, friend_id: str, friend_name: str):
        """使用真实录制注册声纹"""
        from audio_capture import DeviceManager, AudioCapturer
        
        print_title(f"为 {friend_name} 录制声纹样本")
        print_info("需要录制3段语音，每段2-3秒")
        print()
        
        # 获取音频设备管理器
        device_manager = DeviceManager()
        
        # 获取 WASAPI Loopback 设备（用于录制系统/游戏声音）
        loopback_device = device_manager.get_default_wasapi_loopback()
        
        if not loopback_device:
            print_error("未找到 WASAPI Loopback 设备")
            print_warning("此设备用于录制游戏中播放的声音")
            return
        
        # 显示使用的设备
        print_subtitle("音频采集设备")
        print()
        print_info(f"使用设备: {loopback_device['name']}")
        print_info("设备类型: WASAPI Loopback (系统音频回环)")
        print()
        print_warning("注意: 此模式录制的是游戏中播放的声音")
        print("      请在游戏中让好友说话，而不是对着麦克风说话")
        print()
        
        print_warning("录制注意事项：")
        print("  1. 确保游戏音量合适（不要太小）")
        print("  2. 录制时请让游戏中的好友说话")
        print("  3. 尽量减少其他声音干扰（背景音乐、其他玩家等）")
        print("  4. 每段录音让好友说不同的内容")
        print("  5. 建议好友单独说话，避免多人同时发言")
        print()
        
        if not confirm("准备好开始录制了吗？"):
            print_warning("已取消")
            return
        
        audio_segments = []
        
        # 录制3段音频
        for i in range(3):
            print()
            print_subtitle(f"录制第 {i+1} 段语音")
            
            if i == 0:
                print_info("建议内容：让好友介绍自己，例如：“大家好，我是XXX”")
            elif i == 1:
                print_info("建议内容：让好友随意聊天，例如：谈论游戏、天气等")
            else:
                print_info("建议内容：让好友再说一段话，任意内容")
            
            print()
            input("按回车键开始录制...")
            
            # 录制音频
            try:
                print_info("● 正在录制... （录制3秒，请让游戏中的好友开始说话）")
                
                recorded_audio = []
                
                def loopback_callback(audio_data, timestamp):
                    recorded_audio.append(audio_data)
                
                # 创建采集器（使用 WASAPI Loopback 录制游戏声音）
                capturer = AudioCapturer(
                    loopback_device=loopback_device['index'],
                    samplerate=16000,
                    channels=1,
                    chunk_size=480
                )
                capturer.set_loopback_callback(loopback_callback)
                
                capturer.start()
                time.sleep(3)  # 录制3秒
                capturer.stop()
                
                if recorded_audio:
                    audio_segment = np.concatenate(recorded_audio)
                    audio_segments.append(audio_segment)
                    
                    # 计算音量
                    rms = np.sqrt(np.mean(audio_segment ** 2))
                    duration = len(audio_segment) / 16000
                    
                    print_success(f"✓ 录制完成！时长: {duration:.2f}秒，音量: {rms:.4f}")
                    
                    if rms < 0.001:
                        print_warning("警告：音量较小，请确认游戏音量是否正常，或好友是否在说话")
                else:
                    print_error("录制失败：未采集到音频")
                    return
            
            except Exception as e:
                logger.error(f"录制音频失败: {e}", exc_info=True)
                print_error(f"录制失败: {e}")
                return
        
        print()
        print_subtitle("正在注册声纹...")
        
        # 注册声纹
        try:
            success = self.speaker_recognizer.register_speaker(
                friend_id=friend_id,
                name=friend_name,
                audio_segments=audio_segments,
                sample_rate=16000
            )
            
            if success:
                print()
                print_success(f"✓ {friend_name} 声纹注册成功！")
                print(f"好友ID: {friend_id}")
                print(f"样本数: {len(audio_segments)}")
                print(f"总时长: {sum(len(seg)/16000 for seg in audio_segments):.2f}秒")
                print()
                print_info("建议立即进行识别测试验证声纹质量")
            else:
                print_error(f"\n{friend_name} 声纹注册失败")
        
        except Exception as e:
            logger.error(f"声纹注册失败: {e}", exc_info=True)
            print_error(f"注册失败: {e}")
    
    def test_speaker_recognize(self):
        """测试声纹识别"""
        print_subtitle("声纹识别测试")
        
        # 检查是否有已注册的好友
        registered = self.speaker_recognizer.get_registered_speakers()
        
        if not registered:
            print_warning("尚未注册任何好友，请先注册好友")
            return
        
        print(f"当前已注册 {len(registered)} 位好友\n")
        
        # 显示好友列表
        for i, speaker_id in enumerate(registered, 1):
            info = self.speaker_recognizer.get_speaker_info(speaker_id)
            if info:
                print(f"  {i}. {info.name} (ID: {speaker_id})")
        
        print()
        print_info("选择识别模式：")
        print("  1. 使用合成音频测试（快速验证）")
        print("  2. 实时识别游戏语音（游戏内测试）")
        print()
        
        choice = get_user_input("请选择 (1/2)", "2")
        
        if choice == "1":
            self._test_speaker_recognize_synthetic()
        elif choice == "2":
            self._test_speaker_recognize_realtime()
        else:
            print_warning("无效选择，已取消")
    
    def _test_speaker_recognize_synthetic(self):
        """使用合成音频测试识别"""
        print_subtitle("合成音频识别测试")
        
        registered = self.speaker_recognizer.get_registered_speakers()
        first_speaker = registered[0]
        info = self.speaker_recognizer.get_speaker_info(first_speaker)
        
        print(f"生成 {info.name if info else first_speaker} 的测试音频...")
        
        seed = hash(info.name if info else first_speaker) % 1000
        test_audio = generate_test_audio(duration=2.0, seed=seed + 5)
        
        try:
            result = self.speaker_recognizer.recognize(
                audio_segment=test_audio,
                timestamp=time.time(),
                sample_rate=16000
            )
            
            print("\n识别结果:")
            print(f"  是否匹配: {'是' if result.matched else '否'}")
            
            if result.matched:
                matched_info = self.speaker_recognizer.get_speaker_info(result.speaker_id)
                print(f"  识别为: {matched_info.name if matched_info else result.speaker_id}")
                print(f"  置信度: {result.confidence:.3f}")
            
            print(f"  处理时间: {result.processing_time:.2f} ms")
            
            if result.similarity_scores:
                print("\n  相似度分数:")
                for speaker_id, score in result.similarity_scores.items():
                    speaker_info = self.speaker_recognizer.get_speaker_info(speaker_id)
                    name = speaker_info.name if speaker_info else speaker_id
                    print(f"    {name}: {score:.3f}")
        
        except Exception as e:
            logger.error(f"声纹识别失败: {e}", exc_info=True)
            print_error(f"识别失败: {e}")
    
    def _test_speaker_recognize_realtime(self):
        """实时识别游戏语音中的说话人"""
        from audio_capture import DeviceManager, AudioCapturer
        import threading
        import queue
        
        print_title("实时说话人识别")
        print_info("从游戏音频中实时识别正在说话的人")
        print()
        
        # 获取 WASAPI Loopback 设备
        device_manager = DeviceManager()
        loopback_device = device_manager.get_default_wasapi_loopback()
        
        if not loopback_device:
            print_error("未找到 WASAPI Loopback 设备")
            return
        
        print_info(f"使用设备: {loopback_device['name']}")
        print()
        
        # 配置参数
        print_subtitle("测试配置")
        duration = get_number_input("测试时长（秒，0表示手动停止）", 30)
        
        print()
        print_warning("提示：")
        print("  1. 请确保游戏音量合适")
        print("  2. 让已注册的好友在游戏中说话")
        print("  3. 程序将实时显示识别结果")
        print("  4. 按 Ctrl+C 可随时停止")
        print()
        
        if not confirm("准备开始实时识别？"):
            return
        
        # 初始化 VAD（如果还没有）
        if not self.vad_detector:
            from vad import VADDetector
            self.vad_detector = VADDetector(
                sample_rate=16000,
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_ms=10000,
                min_silence_duration_ms=300
            )
        
        # 统计数据
        stats = {
            'total_segments': 0,
            'matched_segments': 0,
            'unknown_segments': 0,
            'speaker_counts': {},
            'start_time': time.time()
        }
        
        # 控制标志
        running = {'flag': True}
        
        # 语音片段队列
        vad_queue = queue.Queue(maxsize=20)
        
        # VAD 回调
        def vad_callback(segment, metadata):
            if running['flag']:
                vad_queue.put((segment, metadata, time.time()))
        
        self.vad_detector.set_callback(vad_callback)
        
        # 音频采集回调
        def audio_callback(audio_data, timestamp):
            if running['flag']:
                try:
                    self.vad_detector.process_audio(audio_data, timestamp)
                except Exception as e:
                    logger.error(f"VAD处理错误: {e}")
        
        # 识别处理线程
        def recognition_thread():
            while running['flag']:
                try:
                    segment, metadata, detect_time = vad_queue.get(timeout=1.0)
                    stats['total_segments'] += 1
                    
                    # 说话人识别
                    try:
                        result = self.speaker_recognizer.recognize(
                            audio_segment=segment,
                            timestamp=detect_time,
                            sample_rate=16000
                        )
                        
                        # 显示识别结果
                        elapsed = time.time() - stats['start_time']
                        
                        if result.matched:
                            stats['matched_segments'] += 1
                            info = self.speaker_recognizer.get_speaker_info(result.speaker_id)
                            speaker_name = info.name if info else result.speaker_id
                            
                            # 统计说话次数
                            if speaker_name not in stats['speaker_counts']:
                                stats['speaker_counts'][speaker_name] = 0
                            stats['speaker_counts'][speaker_name] += 1
                            
                            # 实时显示
                            print(f"\n[{format_duration(elapsed)}] 🎤 {speaker_name}")
                            print(f"  置信度: {result.confidence:.3f} | 时长: {metadata.get('duration', 0):.2f}s")
                            
                            # 显示所有候选人的相似度
                            if result.similarity_scores and len(result.similarity_scores) > 1:
                                print("  其他候选:")
                                sorted_scores = sorted(
                                    result.similarity_scores.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )
                                for speaker_id, score in sorted_scores:
                                    if speaker_id != result.speaker_id:
                                        other_info = self.speaker_recognizer.get_speaker_info(speaker_id)
                                        other_name = other_info.name if other_info else speaker_id
                                        print(f"    {other_name}: {score:.3f}")
                        else:
                            stats['unknown_segments'] += 1
                            print(f"\n[{format_duration(elapsed)}] ❓ 未知说话人")
                            print(f"  时长: {metadata.get('duration', 0):.2f}s")
                            
                            # 显示相似度分数（即使未匹配）
                            if result.similarity_scores:
                                print("  相似度分数:")
                                sorted_scores = sorted(
                                    result.similarity_scores.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )
                                for speaker_id, score in sorted_scores[:3]:  # 显示前3个
                                    other_info = self.speaker_recognizer.get_speaker_info(speaker_id)
                                    other_name = other_info.name if other_info else speaker_id
                                    print(f"    {other_name}: {score:.3f}")
                    
                    except Exception as e:
                        logger.error(f"识别错误: {e}", exc_info=True)
                        print_error(f"识别错误: {e}")
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"处理线程错误: {e}", exc_info=True)
        
        # 创建采集器
        try:
            capturer = AudioCapturer(
                loopback_device=loopback_device['index'],
                samplerate=16000,
                channels=1,
                chunk_size=480
            )
            capturer.set_loopback_callback(audio_callback)
        except Exception as e:
            print_error(f"音频采集器创建失败: {e}")
            return
        
        # 启动识别线程
        recog_thread = threading.Thread(target=recognition_thread, daemon=True)
        recog_thread.start()
        
        # 开始采集
        print_separator()
        print_subtitle("开始实时识别")
        print_info("按 Ctrl+C 停止识别")
        print_separator()
        print()
        
        try:
            capturer.start()
            start_time = time.time()
            
            # 主循环
            while running['flag']:
                time.sleep(0.5)
                
                # 检查时长
                if duration > 0 and (time.time() - start_time) >= duration:
                    print_info("\n测试时长已到，停止识别")
                    break
                
                # 显示简单状态（每5秒）
                if int(time.time() - start_time) % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"\r运行中... {format_duration(elapsed)} | "
                          f"检测片段: {stats['total_segments']} | "
                          f"识别成功: {stats['matched_segments']}",
                          end='', flush=True)
        
        except KeyboardInterrupt:
            print("\n\n用户中断识别")
        finally:
            # 停止采集和处理
            running['flag'] = False
            capturer.stop()
            recog_thread.join(timeout=3)
        
        # 显示统计报告
        print("\n")
        print_separator("=")
        print_title("识别统计报告")
        print_separator("=")
        
        total_time = time.time() - stats['start_time']
        
        print(f"\n测试时长: {format_duration(total_time)}")
        print(f"检测语音片段: {stats['total_segments']}")
        print(f"识别成功: {stats['matched_segments']}")
        print(f"未知说话人: {stats['unknown_segments']}")
        
        if stats['total_segments'] > 0:
            success_rate = stats['matched_segments'] / stats['total_segments'] * 100
            print(f"识别率: {success_rate:.1f}%")
        
        if stats['speaker_counts']:
            print("\n说话统计:")
            sorted_speakers = sorted(
                stats['speaker_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for speaker_name, count in sorted_speakers:
                print(f"  {speaker_name}: {count} 次")
        
        print()
        print_success("✓ 实时识别测试完成！")
    
    def test_speaker_list(self):
        """查看已注册好友"""
        print_subtitle("已注册好友列表")
        
        registered = self.speaker_recognizer.get_registered_speakers()
        
        if not registered:
            print_warning("尚未注册任何好友")
            return
        
        print(f"\n共 {len(registered)} 位好友:\n")
        
        for i, speaker_id in enumerate(registered, 1):
            info = self.speaker_recognizer.get_speaker_info(speaker_id)
            if info:
                print(f"{i}. {info.name}")
                print(f"   ID: {speaker_id}")
                print(f"   样本数: {info.sample_count}")
                print(f"   平均时长: {info.avg_duration:.2f}秒")
                print()
    
    def test_stt(self):
        """测试 STT 模块"""
        print_title("STT 语音转文本测试")
        
        if not self.modules_initialized['stt']:
            print_error("STT 模块未初始化，请先运行系统初始化")
            return
        
        print_info("使用合成音频进行 STT 测试")
        print_warning("注意: 合成音频识别可能失败（这是正常的）")
        print()
        
        try:
            # 生成测试音频
            audio = generate_test_audio(duration=2.0, seed=42)
            
            print_info("开始识别...")
            
            with PerformanceTimer() as timer:
                result = self.stt_recognizer.recognize(audio)
            
            print()
            print_subtitle("识别结果")
            print(f"成功: {result.success}")
            print(f"文本: {result.text}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"语言: {result.language}")
            print(f"引擎: {result.engine_type}")
            print(f"处理时间: {result.processing_time:.1f} ms")
            print(f"总耗时: {timer.elapsed_ms:.1f} ms")
            
            if result.error_message:
                print(f"错误信息: {result.error_message}")
        
        except Exception as e:
            logger.error(f"STT 测试失败: {e}", exc_info=True)
            print_error(f"STT 测试失败: {e}")
    
    def test_memory(self):
        """测试记忆管理模块"""
        print_title("记忆管理测试")
        
        if not self.modules_initialized['memory']:
            print_error("记忆管理模块未初始化，请先运行系统初始化")
            return
        
        options = {
            '1': '创建好友档案',
            '2': '添加对话记录',
            '3': '检索记忆',
            '4': '查看统计信息'
        }
        
        choice = show_menu("记忆管理测试", options)
        
        if choice == '1':
            self.test_memory_create_profile()
        elif choice == '2':
            self.test_memory_add_conversation()
        elif choice == '3':
            self.test_memory_retrieve()
        elif choice == '4':
            self.test_memory_stats()
    
    def test_memory_create_profile(self):
        """测试创建好友档案"""
        print_subtitle("创建好友档案")
        
        name = get_user_input("好友姓名")
        if not name:
            print_warning("已取消")
            return
        
        preferences = get_user_input("偏好话题（逗号分隔）", "游戏,动漫")
        avoid_topics = get_user_input("避免话题（逗号分隔）", "政治")
        personality = get_user_input("性格特点", "活泼")
        
        try:
            friend_id = self.memory_manager.create_friend_profile(
                name=name,
                voice_profile_path=f"data/speaker_profiles/{name}.npy",
                preferences=preferences.split(',') if preferences else [],
                avoid_topics=avoid_topics.split(',') if avoid_topics else [],
                personality=personality
            )
            
            print_success(f"\n好友档案创建成功！")
            print(f"好友ID: {friend_id}")
            print(f"姓名: {name}")
            print(f"偏好: {preferences}")
        
        except Exception as e:
            logger.error(f"创建档案失败: {e}", exc_info=True)
            print_error(f"创建失败: {e}")
    
    def test_memory_add_conversation(self):
        """测试添加对话记录"""
        print_subtitle("添加对话记录")
        
        # 获取现有好友
        try:
            stats = self.memory_manager.get_statistics()
            if stats.get('total_friends', 0) == 0:
                print_warning("尚未创建好友档案，请先创建好友")
                return
            
            friend_id = get_user_input("好友ID（或输入新ID）")
            if not friend_id:
                print_warning("已取消")
                return
            
            text = get_user_input("对话内容")
            if not text:
                print_warning("已取消")
                return
            
            conv_id = self.memory_manager.add_conversation(
                friend_id=friend_id,
                transcript=text,
                speaker_id=friend_id,
                event_type="STATEMENT"
            )
            
            print_success(f"\n对话记录已添加！")
            print(f"对话ID: {conv_id}")
        
        except Exception as e:
            logger.error(f"添加对话失败: {e}", exc_info=True)
            print_error(f"添加失败: {e}")
    
    def test_memory_retrieve(self):
        """测试记忆检索"""
        print_subtitle("记忆检索")
        
        query = get_user_input("检索查询")
        if not query:
            print_warning("已取消")
            return
        
        try:
            memories = self.memory_manager.retrieve_memories(
                query=query,
                top_k=5
            )
            
            print(f"\n找到 {len(memories)} 条相关记忆:\n")
            
            for i, memory in enumerate(memories, 1):
                print(f"{i}. {memory.content[:50]}...")
                print(f"   相似度: {memory.similarity_score:.3f}")
                print(f"   时间衰减: {memory.time_decay_factor:.3f}")
                print(f"   时间: {format_timestamp(memory.timestamp)}")
                print()
        
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            print_error(f"检索失败: {e}")
    
    def test_memory_stats(self):
        """查看记忆管理统计"""
        print_subtitle("记忆管理统计")
        
        try:
            stats = self.memory_manager.get_statistics()
            
            print(f"好友数量: {stats.get('total_friends', 0)}")
            print(f"对话总数: {stats.get('total_conversations', 0)}")
            print(f"向量总数: {stats.get('total_vectors', 0)}")
            print(f"向量维度: {stats.get('embedding_dimension', 0)}")
        
        except Exception as e:
            logger.error(f"获取统计失败: {e}", exc_info=True)
            print_error(f"获取统计失败: {e}")
    
    def run_full_pipeline_test(self):
        """运行完整流程测试 - 选择测试模式"""
        print_title("完整流程测试")
        
        print_warning("此功能需要所有模块都已初始化")
        print_info("将测试: 音频采集 -> VAD -> 说话人识别 -> STT -> 记忆存储")
        print()
        
        # 检查模块状态
        all_ready = all(self.modules_initialized.values())
        
        if not all_ready:
            print_error("部分模块未初始化，无法运行完整流程")
            print_info("请先运行 '系统初始化' 功能")
            return
        
        # 选择测试模式
        options = {
            '1': '实时采集测试 - 从音频设备采集并处理（游戏内测试）',
            '2': '模拟音频测试 - 使用合成音频测试（快速验证）'
        }
        
        choice = show_menu("选择测试模式", options)
        
        if choice == '0':
            return
        elif choice == '1':
            self.run_full_pipeline_test_realtime()
        elif choice == '2':
            self.run_full_pipeline_test_simulated()
    
    def run_full_pipeline_test_simulated(self):
        """运行模拟音频的完整流程测试"""
        print_title("模拟音频完整流程测试")
        
        if not confirm("是否开始模拟测试？"):
            return
        
        print()
        print_subtitle("步骤 1: 生成测试音频")
        test_audio = generate_speech_audio(duration=2.0)
        print_success("测试音频生成完成")
        
        print()
        print_subtitle("步骤 2: VAD 检测")
        # TODO: 实现完整流程
        print_info("VAD 检测... (演示)")
        
        print()
        print_subtitle("步骤 3: 说话人识别")
        print_info("说话人识别... (演示)")
        
        print()
        print_subtitle("步骤 4: 语音转文本")
        print_info("STT 识别... (演示)")
        
        print()
        print_subtitle("步骤 5: 记忆存储与检索")
        print_info("存储对话... (演示)")
        
        print()
        print_success("模拟测试完成！")
    
    def run_full_pipeline_test_realtime(self):
        """运行实时完整流程测试（核心功能）"""
        import threading
        import queue
        import json
        from datetime import datetime
        
        print_title("实时完整流程测试")
        print_info("这将从音频设备实时采集并处理语音")
        print()
        
        # 配置参数
        print_subtitle("测试配置")
        duration = get_number_input("测试时长（秒，0表示手动停止）", 0)
        save_transcripts = confirm("是否保存识别结果？", default=True)
        
        # 检查已注册好友
        registered = self.speaker_recognizer.get_registered_speakers()
        if not registered:
            print_warning("尚未注册任何好友，将跳过说话人识别")
            enable_speaker = False
        else:
            print_info(f"已注册 {len(registered)} 位好友")
            for speaker_id in registered:
                info = self.speaker_recognizer.get_speaker_info(speaker_id)
                if info:
                    print(f"  - {info.name}")
            enable_speaker = True
        
        print()
        if not confirm("确认配置，开始测试？"):
            return
        
        # 初始化统计数据
        stats = {
            'start_time': time.time(),
            'audio_frames': 0,
            'vad_segments': 0,
            'speaker_matches': 0,
            'speaker_unknowns': 0,
            'stt_success': 0,
            'stt_failed': 0,
            'memory_saved': 0,
            'total_audio_duration': 0,
            'results': []
        }
        
        # 创建队列和控制标志
        audio_queue = queue.Queue(maxsize=100)
        vad_queue = queue.Queue(maxsize=20)
        running = {'flag': True}
        paused = {'flag': False}
        
        # VAD回调：检测到语音片段
        def vad_callback(segment, metadata):
            if running['flag'] and not paused['flag']:
                stats['vad_segments'] += 1
                vad_queue.put((segment, metadata))
        
        # 设置VAD回调
        self.vad_detector.set_callback(vad_callback)
        
        # 音频采集回调
        def audio_callback(audio_data, timestamp):
            if running['flag']:
                stats['audio_frames'] += 1
                stats['total_audio_duration'] = len(audio_data) / 16000
                # 将音频数据传递给VAD
                try:
                    self.vad_detector.process_audio(audio_data, timestamp)
                except Exception as e:
                    logger.error(f"VAD处理错误: {e}")
        
        # 获取音频设备
        from audio_capture import DeviceManager, AudioCapturer
        device_manager = DeviceManager()
        loopback = device_manager.get_default_wasapi_loopback()
        
        if not loopback:
            print_error("未找到 WASAPI Loopback 设备")
            return
        
        print_info(f"使用设备: {loopback['name']}")
        print()
        
        # 创建音频采集器
        try:
            capturer = AudioCapturer(
                loopback_device=loopback['index'],
                samplerate=16000,
                channels=1,
                chunk_size=480
            )
            capturer.set_loopback_callback(audio_callback)
        except Exception as e:
            print_error(f"音频采集器创建失败: {e}")
            return
        
        # 处理线程
        def processing_thread():
            """处理VAD检测到的语音片段"""
            while running['flag']:
                try:
                    # 从队列获取语音片段（超时1秒）
                    segment, metadata = vad_queue.get(timeout=1.0)
                    
                    if paused['flag']:
                        continue
                    
                    # 说话人识别
                    speaker_result = None
                    speaker_name = "未知"
                    
                    if enable_speaker:
                        try:
                            speaker_result = self.speaker_recognizer.recognize(
                                audio_segment=segment,
                                timestamp=time.time(),
                                sample_rate=16000
                            )
                            
                            if speaker_result.matched:
                                stats['speaker_matches'] += 1
                                info = self.speaker_recognizer.get_speaker_info(speaker_result.speaker_id)
                                speaker_name = info.name if info else speaker_result.speaker_id
                            else:
                                stats['speaker_unknowns'] += 1
                                speaker_name = "未知说话人"
                        except Exception as e:
                            logger.error(f"说话人识别错误: {e}")
                            stats['speaker_unknowns'] += 1
                    
                    # STT识别（仅对已匹配的说话人）
                    if not enable_speaker or (speaker_result and speaker_result.matched):
                        try:
                            stt_result = self.stt_recognizer.recognize(segment)
                            
                            if stt_result.success and stt_result.text.strip():
                                stats['stt_success'] += 1
                                
                                # 保存到记忆
                                try:
                                    if enable_speaker and speaker_result:
                                        conv_id = self.memory_manager.add_conversation(
                                            friend_id=speaker_result.speaker_id,
                                            transcript=stt_result.text,
                                            speaker_id=speaker_result.speaker_id,
                                            event_type="STATEMENT"
                                        )
                                        stats['memory_saved'] += 1
                                except Exception as e:
                                    logger.error(f"记忆保存错误: {e}")
                                
                                # 记录结果
                                result = {
                                    'timestamp': time.time(),
                                    'speaker': speaker_name,
                                    'confidence': speaker_result.confidence if speaker_result else 0,
                                    'text': stt_result.text,
                                    'stt_confidence': stt_result.confidence,
                                    'duration': metadata.get('duration', 0)
                                }
                                stats['results'].append(result)
                                
                                # 实时显示
                                elapsed = time.time() - stats['start_time']
                                conf = speaker_result.confidence if speaker_result else 0
                                print(f"\n[{format_duration(elapsed)}] {speaker_name} ({conf:.2f}):")
                                print(f"  \"{stt_result.text}\"")
                                print(f"  STT置信度: {stt_result.confidence:.2f} | 时长: {metadata.get('duration', 0):.2f}s")
                            else:
                                stats['stt_failed'] += 1
                        except Exception as e:
                            logger.error(f"STT识别错误: {e}")
                            stats['stt_failed'] += 1
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"处理线程错误: {e}", exc_info=True)
        
        # 启动处理线程
        proc_thread = threading.Thread(target=processing_thread, daemon=True)
        proc_thread.start()
        
        # 启动音频采集
        print_subtitle("开始实时测试")
        print_info("按 Ctrl+C 停止测试")
        print_separator()
        print()
        
        try:
            capturer.start()
            start_time = time.time()
            
            # 主循环
            while running['flag']:
                time.sleep(0.5)
                
                # 检查时长
                if duration > 0 and (time.time() - start_time) >= duration:
                    print_info("\n测试时长已到，停止测试")
                    break
                
                # 显示简单状态（每5秒）
                if int(time.time() - start_time) % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"\r运行中... {format_duration(elapsed)} | "
                          f"VAD片段: {stats['vad_segments']} | "
                          f"识别成功: {stats['stt_success']}", end='', flush=True)
        
        except KeyboardInterrupt:
            print("\n\n用户中断测试")
        finally:
            # 停止采集和处理
            running['flag'] = False
            capturer.stop()
            proc_thread.join(timeout=3)
        
        # 显示统计报告
        print("\n")
        print_separator("=")
        print_title("测试完成 - 统计报告")
        print_separator("=")
        
        total_time = time.time() - stats['start_time']
        
        print(f"\n测试时长: {format_duration(total_time)}")
        print(f"音频帧数: {stats['audio_frames']}")
        print(f"VAD检测片段: {stats['vad_segments']}")
        
        if enable_speaker:
            print(f"\n说话人识别:")
            print(f"  匹配成功: {stats['speaker_matches']}")
            print(f"  未知说话人: {stats['speaker_unknowns']}")
            if stats['speaker_matches'] + stats['speaker_unknowns'] > 0:
                match_rate = stats['speaker_matches'] / (stats['speaker_matches'] + stats['speaker_unknowns']) * 100
                print(f"  匹配率: {match_rate:.1f}%")
        
        print(f"\nSTT识别:")
        print(f"  成功: {stats['stt_success']}")
        print(f"  失败: {stats['stt_failed']}")
        if stats['stt_success'] + stats['stt_failed'] > 0:
            success_rate = stats['stt_success'] / (stats['stt_success'] + stats['stt_failed']) * 100
            print(f"  成功率: {success_rate:.1f}%")
        
        print(f"\n记忆管理:")
        print(f"  保存对话: {stats['memory_saved']} 条")
        
        # 保存结果
        if save_transcripts and stats['results']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            transcript_file = project_root / f"tests/integrated_test_{timestamp}_transcripts.txt"
            report_file = project_root / f"tests/integrated_test_{timestamp}_report.json"
            
            # 保存文本转录
            try:
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(f"VRChat 社交辅助工具 - 完整流程测试结果\n")
                    f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"测试时长: {format_duration(total_time)}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for i, result in enumerate(stats['results'], 1):
                        elapsed = result['timestamp'] - stats['start_time']
                        f.write(f"[{format_duration(elapsed)}] {result['speaker']} ({result['confidence']:.2f}):\n")
                        f.write(f"  \"{result['text']}\"\n")
                        f.write(f"  STT置信度: {result['stt_confidence']:.2f} | 时长: {result['duration']:.2f}s\n\n")
                
                print(f"\n转录结果已保存: {transcript_file.name}")
            except Exception as e:
                logger.error(f"保存转录失败: {e}")
            
            # 保存JSON报告
            try:
                report = {
                    'test_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'duration_seconds': total_time,
                        'test_mode': 'realtime'
                    },
                    'module_statistics': {
                        'audio_frames': stats['audio_frames'],
                        'vad_segments': stats['vad_segments'],
                        'speaker_matches': stats['speaker_matches'],
                        'speaker_unknowns': stats['speaker_unknowns'],
                        'stt_success': stats['stt_success'],
                        'stt_failed': stats['stt_failed'],
                        'memory_saved': stats['memory_saved']
                    },
                    'results': stats['results']
                }
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                print(f"测试报告已保存: {report_file.name}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        print()
        print_success("✓ 实时完整流程测试完成！")
    
    def show_data_management_menu(self):
        """显示数据管理菜单"""
        while True:
            options = {
                '1': '好友管理',
                '2': '对话记录查看',
                '3': '数据统计',
                '4': '数据清理'
            }
            
            choice = show_menu("数据管理", options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.manage_friends()
            elif choice == '2':
                self.view_conversations()
            elif choice == '3':
                self.show_data_stats()
            elif choice == '4':
                self.cleanup_data()
            
            if choice != '0':
                wait_for_enter()
    
    def manage_friends(self):
        """好友管理"""
        print_title("好友管理")
        print_info("此功能整合了说话人识别和记忆管理的好友数据")
    
    def view_conversations(self):
        """查看对话记录"""
        print_title("对话记录")
        print_info("显示历史对话记录")
    
    def show_data_stats(self):
        """显示数据统计"""
        print_title("数据统计")
        
        # 显示各模块的统计信息
        if self.modules_initialized['speaker']:
            print_subtitle("说话人识别", "-")
            registered = self.speaker_recognizer.get_registered_speakers()
            print(f"已注册好友: {len(registered)} 位")
            print()
        
        if self.modules_initialized['memory']:
            print_subtitle("记忆管理", "-")
            try:
                stats = self.memory_manager.get_statistics()
                print(f"好友档案: {stats.get('total_friends', 0)} 个")
                print(f"对话记录: {stats.get('total_conversations', 0)} 条")
                print(f"向量数据: {stats.get('total_vectors', 0)} 条")
            except:
                print_warning("无法获取统计信息")
    
    def cleanup_data(self):
        """数据清理"""
        print_title("数据清理")
        print_warning("此操作将删除数据，请谨慎操作！")
        
        if not confirm("确定要清理数据吗？", default=False):
            print_info("已取消")
            return
        
        print_info("数据清理功能尚未实现")
    
    def show_system_info(self):
        """显示系统信息"""
        print_title("系统信息")
        
        print_subtitle("模块状态", "-")
        for module, status in self.modules_initialized.items():
            status_text = "✓ 已初始化" if status else "✗ 未初始化"
            if status:
                print_success(f"{module.upper()}: {status_text}")
            else:
                print_error(f"{module.upper()}: {status_text}")
        
        print()
        print_subtitle("环境信息", "-")
        print(f"Python 版本: {sys.version.split()[0]}")
        print(f"项目路径: {project_root}")
        print(f"配置目录: {self.config_dir}")
        print(f"数据目录: {self.data_dir}")
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理资源...")
        
        if self.audio_capturer:
            try:
                self.audio_capturer.stop()
            except:
                pass
        
        logger.info("资源清理完成")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='VRChat 社交辅助工具 - 功能集成测试程序',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--init', action='store_true',
                       help='仅运行初始化检查')
    parser.add_argument('--module', type=str,
                       help='直接进入指定模块测试 (audio/vad/speaker/stt/memory)')
    parser.add_argument('--full', action='store_true',
                       help='直接运行完整流程测试')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 设置环境变量避免 PyTorch 死锁问题（Windows）
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # 创建测试实例
    test = IntegratedTest()
    
    # 根据参数执行
    if args.init:
        test.initialize_system()
    elif args.module:
        test.initialize_system()
        # TODO: 根据参数直接进入对应模块测试
        print_info(f"进入 {args.module} 模块测试...")
    elif args.full:
        test.initialize_system()
        test.run_full_pipeline_test()
    else:
        # 正常启动
        test.run()


if __name__ == '__main__':
    # Windows 多进程保护
    import multiprocessing
    multiprocessing.freeze_support()
    
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warning("用户中断操作")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序异常退出: {e}", exc_info=True)
        print_error(f"程序异常: {e}")
        sys.exit(1)
