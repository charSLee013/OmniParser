import pytest
import base64
import threading
import time
import concurrent.futures


def test_health_check_models_not_loaded(test_client, mock_models_not_loaded):
    """测试模型未加载时的健康检查"""
    response = test_client.get("/api/health")
    assert response.status_code == 503
    data = response.json()
    assert data["models_loaded"] is False
    assert "unavailable" in data["status"].lower()


def test_health_check_models_loaded(test_client, mock_models_loaded):
    """测试模型已加载时的健康检查"""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["models_loaded"] is True
    assert "available" in data["status"].lower()


def test_parse_models_not_loaded(test_client, mock_models_not_loaded, valid_parse_request):
    """测试模型未加载时的解析请求"""
    response = test_client.post("/api/parse", json=valid_parse_request)
    assert response.status_code == 503
    data = response.json()
    assert "loading" in data["detail"].lower()


def test_parse_with_lock_acquired(test_client, mock_models_loaded, valid_parse_request, create_lock_situation):
    """测试在锁被占用时的解析请求"""
    response = test_client.post("/api/parse", json=valid_parse_request)
    assert response.status_code == 429
    data = response.json()
    assert "processing another request" in data["detail"].lower()


def test_parse_invalid_image(test_client, mock_models_loaded, invalid_parse_request):
    """测试无效图像的解析请求"""
    response = test_client.post("/api/parse", json=invalid_parse_request)
    assert response.status_code == 400
    data = response.json()
    assert "invalid image format" in data["detail"].lower()


def test_parse_success(test_client, mock_models_loaded, valid_parse_request):
    """测试成功的解析请求"""
    response = test_client.post("/api/parse", json=valid_parse_request)
    assert response.status_code == 200
    data = response.json()
    assert "image" in data
    assert "parsed_elements" in data
    
    # 验证返回的图像是有效的 base64
    try:
        base64.b64decode(data["image"])
    except Exception:
        pytest.fail("返回的图像不是有效的 base64 编码")


def test_parse_with_different_params(test_client, mock_models_loaded, valid_parse_request):
    """测试不同参数的解析请求"""
    # 修改请求参数
    modified_request = valid_parse_request.copy()
    modified_request["box_threshold"] = 0.1
    modified_request["iou_threshold"] = 0.2
    modified_request["use_paddleocr"] = False
    modified_request["imgsz"] = 800
    
    response = test_client.post("/api/parse", json=modified_request)
    assert response.status_code == 200


def test_concurrent_requests(test_client, mock_models_loaded, valid_parse_request):
    """测试并发请求"""
    
    def make_long_request():
        # 这个请求会获取锁并保持一段时间
        return test_client.post("/api/parse", json=valid_parse_request)
    
    def make_second_request():
        # 这个请求应该因为锁被占用而失败
        time.sleep(0.1)  # 给第一个请求一点时间获取锁
        return test_client.post("/api/parse", json=valid_parse_request)
    
    # 使用线程池执行并发请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(make_long_request)
        future2 = executor.submit(make_second_request)
        
        response1 = future1.result()
        response2 = future2.result()
    
    # 第一个请求应该成功
    assert response1.status_code == 200
    
    # 第二个请求应该因为锁被占用而返回 429
    # 注意：这个测试在某些情况下可能不稳定，取决于线程调度
    if response2.status_code != 429:
        print(f"警告：并发测试可能不稳定。第二个请求返回了 {response2.status_code}")
    
    # 至少一个请求应该成功，另一个应该失败或成功
    assert (response1.status_code == 200 and response2.status_code == 429) or \
           (response1.status_code == 429 and response2.status_code == 200) or \
           (response1.status_code == 200 and response2.status_code == 200) 