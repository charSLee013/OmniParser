import base64
import os
import sys
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import threading
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入 API 服务器
import api_server


@pytest.fixture
def test_client():
    """创建测试客户端"""
    client = TestClient(api_server.app)
    return client


@pytest.fixture
def test_image_base64():
    """将测试图片转换为 base64 字符串"""
    image_path = "imgs/windows.png"
    if not os.path.exists(image_path):
        pytest.skip(f"测试图片不存在: {image_path}")
    
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        base64_str = base64.b64encode(img_data).decode('utf-8')
    return base64_str


@pytest.fixture
def mock_models_not_loaded(monkeypatch):
    """模拟模型未加载的状态"""
    monkeypatch.setattr(api_server, "models_loaded", False)
    monkeypatch.setattr(api_server, "model_loading", False)


@pytest.fixture
def mock_models_loaded(monkeypatch):
    """模拟模型已加载的状态"""
    monkeypatch.setattr(api_server, "models_loaded", True)
    monkeypatch.setattr(api_server, "model_loading", False)
    
    # 为测试创建一个伪造的模型
    class MockModel:
        def predict(self, *args, **kwargs):
            # 返回类似于 YOLO 模型输出的结构
            class BoxResult:
                def __init__(self):
                    self.xyxy = []
                    self.conf = []
            
            class Result:
                def __init__(self):
                    self.boxes = BoxResult()
            
            return [Result()]
    
    # 模拟 caption_model_processor
    caption_processor = {
        'model': MockModel(),
        'processor': {}
    }
    
    # 替换 get_yolo_model 和 get_som_labeled_img 的功能
    def mock_check_ocr_box(*args, **kwargs):
        return ([], []), None
    
    def mock_get_som_labeled_img(*args, **kwargs):
        # 返回一个空白图像的 base64 编码
        img = Image.new('RGB', (100, 100), color = 'white')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str, {}, []
    
    monkeypatch.setattr(api_server, "yolo_model", MockModel())
    monkeypatch.setattr(api_server, "caption_model_processor", caption_processor)
    monkeypatch.setattr(api_server, "check_ocr_box", mock_check_ocr_box)
    monkeypatch.setattr(api_server, "get_som_labeled_img", mock_get_som_labeled_img)


@pytest.fixture
def create_lock_situation():
    """创建一个锁占用的情况"""
    api_server.process_lock.acquire()
    yield
    # 确保在测试后释放锁
    if api_server.process_lock.locked():
        api_server.process_lock.release()


@pytest.fixture
def valid_parse_request(test_image_base64):
    """创建有效的解析请求数据"""
    return {
        "image": test_image_base64,
        "box_threshold": 0.05,
        "iou_threshold": 0.1,
        "use_paddleocr": True,
        "imgsz": 640
    }


@pytest.fixture
def invalid_parse_request():
    """创建无效的解析请求数据"""
    return {
        "image": "invalid_base64_data",
        "box_threshold": 0.05,
        "iou_threshold": 0.1,
        "use_paddleocr": True,
        "imgsz": 640
    } 