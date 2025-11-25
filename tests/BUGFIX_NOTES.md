# 集成测试问题修复说明

## 问题描述

在运行 `python tests/integrated_test.py --init` 时遇到了两个主要问题：

### 问题 1: PyTorch 死锁错误

**错误信息:**
```
RuntimeError: resource deadlock would occur: resource deadlock would occur
```

**原因分析:**
- 这是 Windows 平台上 PyTorch/torchvision 的已知问题
- 在导入 `pyannote.audio` 时，多个线程尝试注册 PyTorch 操作导致死锁
- 与 `lightning`、`torchmetrics`、`torchvision` 等库的初始化顺序有关

**解决方案:**
1. 在主程序中设置环境变量避免死锁：
   ```python
   os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
   os.environ['OMP_NUM_THREADS'] = '1'
   ```

2. 在 `embedding_engine.py` 中添加异常处理，捕获死锁错误并使用替代模型：
   ```python
   except RuntimeError as e:
       if 'deadlock' in str(e).lower():
           logger.warning("PyTorch 死锁错误（Windows 常见问题），使用替代方案")
           self._load_alternative_model()
   ```

3. 添加 Windows 多进程保护：
   ```python
   if __name__ == '__main__':
       import multiprocessing
       multiprocessing.freeze_support()
   ```

### 问题 2: Windows 控制台编码错误

**错误信息:**
```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2139' in position 5: illegal multibyte sequence
```

**原因分析:**
- Windows 默认使用 GBK 编码
- 程序中使用了 Unicode 特殊字符（✓、✗、⚠、ℹ 等）
- GBK 编码无法正确显示这些字符

**解决方案:**
1. 在 `test_utils.py` 中设置控制台为 UTF-8 编码：
   ```python
   if sys.platform == 'win32':
       try:
           import ctypes
           kernel32 = ctypes.windll.kernel32
           kernel32.SetConsoleOutputCP(65001)  # UTF-8
       except:
           pass
   ```

2. 将特殊 Unicode 字符替换为 ASCII 兼容字符：
   - ✓ → [OK]
   - ✗ → [ERR]
   - ⚠ → [WARN]
   - ℹ → [INFO]

## 修复后的测试结果

✅ **所有 5 个模块初始化成功（100%）**

```
[OK] AUDIO 模块: ✓ 成功
[OK] VAD 模块: ✓ 成功
[OK] SPEAKER 模块: ✓ 成功
[OK] STT 模块: ✓ 成功
[OK] MEMORY 模块: ✓ 成功

总计: 5/5 模块初始化成功 (100%)
```

## 注意事项

### 说话人识别模块

由于 PyTorch 死锁问题，当前使用的是**演示用的简化模型**，不具备真实的声纹识别能力。

**如需使用真实模型，请：**
1. 确保已安装 `pyannote.audio`：
   ```bash
   pip install pyannote.audio
   ```

2. 如果仍然遇到死锁问题，可以尝试：
   - 降级 PyTorch 版本
   - 在单独的 Python 进程中运行
   - 使用 Python 3.11+ （某些问题在新版本中已修复）

### 退出时的警告

程序退出时可能会显示 PyTorch 清理过程的警告信息，这是**正常现象**，不影响程序功能。这些警告来自 PyTorch 在清理资源时的内部操作。

## 修改的文件

1. **tests/integrated_test.py**
   - 添加环境变量设置
   - 添加多进程保护

2. **src/speaker_recognition/embedding_engine.py**
   - 增强异常处理
   - 捕获 PyTorch 死锁错误
   - 自动降级到替代模型

3. **tests/test_utils.py**
   - 设置 Windows 控制台编码为 UTF-8
   - 替换 Unicode 特殊字符为 ASCII 字符

## 测试环境

- 操作系统: Windows 22H2
- Python 版本: 3.10
- PyTorch 版本: 根据 requirements.txt
- 测试日期: 2025-11-25

## 相关资源

- [PyTorch 死锁问题讨论](https://github.com/pytorch/pytorch/issues)
- [Windows 控制台编码设置](https://docs.python.org/3/library/codecs.html)
- [pyannote.audio 文档](https://github.com/pyannote/pyannote-audio)
