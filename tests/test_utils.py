"""
集成测试工具函数模块

提供集成测试所需的工具函数，包括：
- 控制台格式化输出
- 用户输入处理
- 测试数据生成
- 性能统计
"""

import sys
import time
import numpy as np
from typing import Optional, List, Callable, Any, Dict
from datetime import datetime

# 设置 Windows 控制台编码为 UTF-8
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # 启用 UTF-8 输出
        kernel32.SetConsoleOutputCP(65001)
    except:
        pass


class ColorText:
    """彩色文本输出（Windows 兼容）"""
    
    # ANSI 颜色代码
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def enable_windows_color():
        """启用 Windows 控制台颜色支持"""
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                pass
    
    @classmethod
    def success(cls, text: str) -> str:
        """成功消息（绿色）"""
        return f"{cls.OKGREEN}{text}{cls.ENDC}"
    
    @classmethod
    def error(cls, text: str) -> str:
        """错误消息（红色）"""
        return f"{cls.FAIL}{text}{cls.ENDC}"
    
    @classmethod
    def warning(cls, text: str) -> str:
        """警告消息（黄色）"""
        return f"{cls.WARNING}{text}{cls.ENDC}"
    
    @classmethod
    def info(cls, text: str) -> str:
        """信息消息（蓝色）"""
        return f"{cls.OKBLUE}{text}{cls.ENDC}"
    
    @classmethod
    def header(cls, text: str) -> str:
        """标题消息（粗体）"""
        return f"{cls.BOLD}{text}{cls.ENDC}"


# 初始化 Windows 颜色支持
ColorText.enable_windows_color()


def print_separator(char: str = "=", length: int = 70):
    """打印分隔线"""
    print(char * length)


def print_title(title: str, char: str = "="):
    """打印标题"""
    print_separator(char)
    print(ColorText.header(title.center(70)))
    print_separator(char)


def print_subtitle(title: str, char: str = "-"):
    """打印子标题"""
    print_separator(char, 50)
    print(ColorText.header(title))
    print_separator(char, 50)


def print_success(message: str, prefix: str = "[OK]"):
    """打印成功消息"""
    print(ColorText.success(f"{prefix} {message}"))


def print_error(message: str, prefix: str = "[ERR]"):
    """打印错误消息"""
    print(ColorText.error(f"{prefix} {message}"))


def print_warning(message: str, prefix: str = "[WARN]"):
    """打印警告消息"""
    print(ColorText.warning(f"{prefix} {message}"))


def print_info(message: str, prefix: str = "[INFO]"):
    """打印信息消息"""
    print(ColorText.info(f"{prefix} {message}"))


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """
    获取用户输入
    
    Args:
        prompt: 提示信息
        default: 默认值（可选）
    
    Returns:
        用户输入的字符串
    """
    if default is not None:
        prompt = f"{prompt} (默认: {default}): "
    else:
        prompt = f"{prompt}: "
    
    user_input = input(prompt).strip()
    
    if not user_input and default is not None:
        return default
    
    return user_input


def get_choice(prompt: str, valid_choices: List[str]) -> str:
    """
    获取用户选择
    
    Args:
        prompt: 提示信息
        valid_choices: 有效选项列表
    
    Returns:
        用户选择的选项
    """
    while True:
        choice = input(f"{prompt} [{'/'.join(valid_choices)}]: ").strip()
        if choice in valid_choices:
            return choice
        print_warning(f"无效的选择，请选择: {', '.join(valid_choices)}")


def confirm(prompt: str, default: bool = False) -> bool:
    """
    获取用户确认
    
    Args:
        prompt: 提示信息
        default: 默认值
    
    Returns:
        用户确认结果
    """
    default_str = "Y/n" if default else "y/N"
    choice = input(f"{prompt} [{default_str}]: ").strip().lower()
    
    if not choice:
        return default
    
    return choice in ['y', 'yes', '是', 'ok']


def get_number_input(prompt: str, min_val: Optional[float] = None, 
                     max_val: Optional[float] = None, 
                     default: Optional[float] = None) -> float:
    """
    获取数字输入
    
    Args:
        prompt: 提示信息
        min_val: 最小值
        max_val: 最大值
        default: 默认值
    
    Returns:
        用户输入的数字
    """
    while True:
        try:
            user_input = get_user_input(prompt, str(default) if default is not None else None)
            value = float(user_input)
            
            if min_val is not None and value < min_val:
                print_warning(f"输入值不能小于 {min_val}")
                continue
            
            if max_val is not None and value > max_val:
                print_warning(f"输入值不能大于 {max_val}")
                continue
            
            return value
        
        except ValueError:
            print_warning("请输入有效的数字")


def show_menu(title: str, options: Dict[str, str], show_exit: bool = True) -> str:
    """
    显示菜单并获取选择
    
    Args:
        title: 菜单标题
        options: 选项字典 {编号: 描述}
        show_exit: 是否显示退出选项
    
    Returns:
        用户选择的选项编号
    """
    print_title(title)
    print()
    
    for key, desc in options.items():
        print(f"  [{key}] {desc}")
    
    if show_exit:
        print(f"  [0] 返回上级/退出")
    
    print()
    
    valid_choices = list(options.keys())
    if show_exit:
        valid_choices.append('0')
    
    return get_choice("请选择", valid_choices)


def generate_test_audio(duration: float = 2.0, 
                       sample_rate: int = 16000, 
                       seed: Optional[int] = None) -> np.ndarray:
    """
    生成测试音频（正弦波组合）
    
    Args:
        duration: 时长（秒）
        sample_rate: 采样率
        seed: 随机种子
    
    Returns:
        音频数据（numpy array）
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    
    # 基频（模拟不同说话人的音高）
    f0 = 150 + (seed or 0) * 20
    
    # 生成信号
    signal = np.sin(2 * np.pi * f0 * t)
    signal += 0.5 * np.sin(2 * np.pi * f0 * 2 * t)  # 二次谐波
    signal += 0.3 * np.random.randn(num_samples)  # 噪声
    
    # 归一化
    signal = signal / np.max(np.abs(signal)) * 0.3
    
    return signal.astype(np.float32)


def generate_speech_audio(duration: float = 1.0, 
                         sample_rate: int = 16000) -> np.ndarray:
    """
    生成模拟语音的测试音频
    
    Args:
        duration: 时长（秒）
        sample_rate: 采样率
    
    Returns:
        音频数据
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    
    # 混合多个频率模拟语音
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 880 * t) +  # A5
        0.1 * np.sin(2 * np.pi * 220 * t)    # A3
    )
    
    return signal.astype(np.float32)


def generate_silence(duration: float = 0.5, 
                    sample_rate: int = 16000) -> np.ndarray:
    """
    生成静音
    
    Args:
        duration: 时长（秒）
        sample_rate: 采样率
    
    Returns:
        音频数据
    """
    num_samples = int(duration * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        if self.start_time is not None:
            self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()
        if self.name:
            print(f"{self.name}: {self.elapsed_ms:.2f} ms")
        return False
    
    def get_elapsed_ms(self) -> float:
        """获取经过的毫秒数"""
        return self.elapsed_ms


class StatisticsCollector:
    """统计信息收集器"""
    
    def __init__(self):
        self.data: Dict[str, List[float]] = {}
    
    def add(self, name: str, value: float):
        """添加数据点"""
        if name not in self.data:
            self.data[name] = []
        self.data[name].append(value)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取统计信息"""
        if name not in self.data or not self.data[name]:
            return {}
        
        values = self.data[name]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'total': sum(values)
        }
    
    def print_summary(self):
        """打印统计摘要"""
        print_subtitle("性能统计摘要")
        for name, values in self.data.items():
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  次数: {stats['count']}")
            print(f"  最小值: {stats['min']:.2f} ms")
            print(f"  最大值: {stats['max']:.2f} ms")
            print(f"  平均值: {stats['mean']:.2f} ms")
            print(f"  总计: {stats['total']:.2f} ms")


def wait_for_enter(prompt: str = "按回车键继续..."):
    """等待用户按回车"""
    input(f"\n{prompt}")


def clear_screen():
    """清空屏幕（跨平台）"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_progress(current: int, total: int, prefix: str = "", 
                  bar_length: int = 40):
    """
    打印进度条
    
    Args:
        current: 当前进度
        total: 总数
        prefix: 前缀文本
        bar_length: 进度条长度
    """
    if total == 0:
        return
    
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent*100:.1f}% ({current}/{total})', end='')
    
    if current >= total:
        print()  # 完成后换行


def format_timestamp(timestamp: float) -> str:
    """
    格式化时间戳
    
    Args:
        timestamp: Unix 时间戳
    
    Returns:
        格式化的时间字符串
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """
    格式化时长
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时长字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分"


def format_size(bytes_size: int) -> str:
    """
    格式化文件大小
    
    Args:
        bytes_size: 字节数
    
    Returns:
        格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def safe_execute(func: Callable, error_message: str = "操作失败", 
                 return_on_error: Any = None) -> Any:
    """
    安全执行函数，捕获异常
    
    Args:
        func: 要执行的函数
        error_message: 错误消息
        return_on_error: 出错时的返回值
    
    Returns:
        函数执行结果或错误返回值
    """
    try:
        return func()
    except Exception as e:
        print_error(f"{error_message}: {str(e)}")
        return return_on_error


def print_table(headers: List[str], rows: List[List[str]], 
                col_widths: Optional[List[int]] = None):
    """
    打印表格
    
    Args:
        headers: 表头
        rows: 数据行
        col_widths: 列宽（可选）
    """
    if not col_widths:
        # 自动计算列宽
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)
    
    # 打印表头
    header_line = "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * sum(col_widths))
    
    # 打印数据行
    for row in rows:
        row_line = "".join(str(row[i]).ljust(col_widths[i]) if i < len(row) else " " * col_widths[i] 
                          for i in range(len(headers)))
        print(row_line)
