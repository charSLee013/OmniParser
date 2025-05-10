import pytest
import os
import sys

if __name__ == "__main__":
    # 添加项目根目录到 Python 路径
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # 运行测试
    exit_code = pytest.main(["-xvs", "tests/"])
    
    # 根据测试结果退出
    sys.exit(exit_code) 