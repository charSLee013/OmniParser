import io
import base64
import threading
import time
import os
import logging
from typing import Optional, Dict, List, Any, Union
from contextlib import contextmanager

import torch
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("omniparser_api.log"),
    ]
)
logger = logging.getLogger("omniparser-api")

# 内存监控函数
def check_memory_usage() -> Dict[str, float]:
    """检查当前内存使用情况并返回信息"""
    memory_info = {}
    
    try:
        # 检查系统内存
        import psutil
        process = psutil.Process(os.getpid())
        memory_info["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
        memory_info["system_available_mb"] = psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        memory_info["process_memory_mb"] = -1
        memory_info["system_available_mb"] = -1
    
    # 检查 CUDA 内存
    if torch.cuda.is_available():
        memory_info["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory_info["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        memory_info["cuda_max_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return memory_info

# 安全内存阈值
MAX_MEMORY_USAGE_MB = 8000  # 8GB
MAX_CUDA_MEMORY_USAGE_MB = 4000  # 4GB

def is_memory_safe():
    """检查内存使用是否在安全范围内"""
    memory_info = check_memory_usage()
    
    # 检查系统内存
    if memory_info.get("process_memory_mb", 0) > MAX_MEMORY_USAGE_MB:
        return False
    
    # 检查 CUDA 内存
    if torch.cuda.is_available() and memory_info.get("cuda_allocated_mb", 0) > MAX_CUDA_MEMORY_USAGE_MB:
        return False
    
    return True

# 全局变量
models_loaded = False
model_loading = False
yolo_model = None
caption_model_processor = None
process_lock = threading.Lock()

# 加载模型函数
def load_models():
    global yolo_model, caption_model_processor, models_loaded, model_loading
    
    if models_loaded or model_loading:
        return
    
    model_loading = True
    try:
        logger.info("Loading models...")
        
        # 记录加载前的内存状态
        logger.info(f"Memory before model loading: {check_memory_usage()}")
        
        # 加载YOLO模型
        yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
        
        # 使用 torch.no_grad() 加载描述模型
        with torch.no_grad():
            caption_model_processor = get_caption_model_processor(
                model_name="florence2", 
                model_name_or_path="Florence-2-base"
            )
        
        # 清理加载过程中的缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 记录加载后的内存状态
        logger.info(f"Memory after model loading: {check_memory_usage()}")
        
        models_loaded = True
        logger.info("Models loaded successfully")
    except Exception as e:
        # Start Generation Here
        logger.error("Failed to load models, full stack trace:", exc_info=True)
        # End Generation Her
        logger.error(f"Failed to load models: {e}")
        models_loaded = False
    finally:
        model_loading = False

# 在创建 FastAPI 应用前同步加载模型
logger.info("Starting to load models synchronously...")
load_models()

# 初始化 FastAPI 应用
app = FastAPI(
    title="OmniParser API",
    description="API for parsing GUI screens into structured elements",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class ParseRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    box_threshold: float = Field(0.05, description="Box threshold for removing low confidence bounding boxes")
    iou_threshold: float = Field(0.1, description="IOU threshold for removing overlapping bounding boxes")
    use_paddleocr: bool = Field(True, description="Whether to use PaddleOCR instead of EasyOCR")
    imgsz: int = Field(640, description="Image size for icon detection", ge=640, le=1920)

# 响应模型
class ParseResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded image with annotations")
    parsed_elements: str = Field(..., description="Parsed screen elements")

# 健康检查响应模型
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    timestamp: float
    memory_info: Dict[str, float] = Field({}, description="Memory usage information")

# 辅助函数：将 base64 字符串转换为 PIL 图像
def base64_to_pil(image_base64: str) -> Image.Image:
    """
    将 base64 编码的图像字符串转换为 PIL Image 对象，优化内存使用
    """
    try:
        if "base64," in image_base64:
            # 处理 "data:image/jpeg;base64," 格式
            image_base64 = image_base64.split("base64,")[1]
        
        # 添加更严格的 base64 验证
        if not isinstance(image_base64, str) or len(image_base64.strip()) == 0:
            raise ValueError("Empty or invalid base64 string")
        
        # 尝试解码 base64 字符串
        try:
            image_data = base64.b64decode(image_base64)
            # 立即释放 base64 字符串内存，减少峰值内存使用
            del image_base64
        except Exception as e:
            raise ValueError(f"Base64 decoding error: {str(e)}")
        
        # 尝试打开图像，减少验证步骤以降低内存使用
        try:
            # 直接打开图像，跳过验证步骤
            image = Image.open(io.BytesIO(image_data))
            # 立即释放图像数据内存
            del image_data
            return image
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
            
    except Exception as e:
        # 捕获所有异常，并以 400 Bad Request 抛出
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )

# 定期清理内存的后台任务
@app.on_event("startup")
async def setup_periodic_tasks():
    import asyncio
    
    async def periodic_memory_cleanup():
        while True:
            # 等待5分钟
            await asyncio.sleep(300)
            
            logger.info("Performing periodic memory cleanup")
            
            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 触发垃圾回收
            import gc
            gc.collect()
            
            logger.info(f"Memory after cleanup: {check_memory_usage()}")
    
    # 启动后台任务
    asyncio.create_task(periodic_memory_cleanup())

# 健康检查端点
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    memory_info = check_memory_usage()
    
    if not models_loaded:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        status_message = "Service unavailable - Models not loaded"
    else:
        status_code = status.HTTP_200_OK
        status_message = "Service available"
    
    response = HealthResponse(
        status=status_message,
        models_loaded=models_loaded,
        timestamp=time.time(),
        memory_info=memory_info
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump()
    )

# 图像解析端点
@app.post("/api/parse", response_model=ParseResponse)
def parse_image(request: ParseRequest):
    # 检查模型是否已加载
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading, please try again later"
        )
    
    # 检查内存使用是否安全
    if not is_memory_safe():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server memory usage is too high, please try again later"
        )
    
    # 尝试获取处理锁
    if not process_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is currently processing another request, please try again later"
        )
    
    # 记录请求开始时的内存状态
    request_id = f"req-{time.time()}"
    logger.info(f"[{request_id}] Starting image processing")
    logger.info(f"[{request_id}] Memory at start: {check_memory_usage()}")
    
    try:
        # 将 base64 图像转换为 PIL Image
        image_input = base64_to_pil(request.image)
        
        # 限制图像大小
        MAX_IMAGE_SIZE = (3000, 3000)
        if image_input.width > MAX_IMAGE_SIZE[0] or image_input.height > MAX_IMAGE_SIZE[1]:
            logger.info(f"[{request_id}] Resizing large image from {image_input.width}x{image_input.height} to fit within {MAX_IMAGE_SIZE}")
            # 等比例缩小图像
            image_input.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
        
        # 释放请求中的base64图像数据，减少内存使用
        del request.image
        
        logger.info(f"[{request_id}] Memory after image decode: {check_memory_usage()}")
        
        # 准备图像处理参数
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        # 使用 torch.no_grad() 上下文来禁用梯度计算，减少内存使用
        with torch.no_grad():
            # 执行 OCR
            logger.info(f"[{request_id}] Starting OCR processing")
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                image_input, 
                display_img=False, 
                output_bb_format='xyxy', 
                goal_filtering=None, 
                easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
                use_paddleocr=request.use_paddleocr
            )
            text, ocr_bbox = ocr_bbox_rslt
            
            # 清理不再需要的中间变量
            del ocr_bbox_rslt, is_goal_filtered
            
            logger.info(f"[{request_id}] Memory after OCR: {check_memory_usage()}")
            
            # 获取标记后的图像和解析结果
            logger.info(f"[{request_id}] Starting icon detection and captioning")
            dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                image_input, 
                yolo_model, 
                BOX_TRESHOLD=request.box_threshold, 
                output_coord_in_ratio=True, 
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config, 
                caption_model_processor=caption_model_processor, 
                ocr_text=text,
                iou_threshold=request.iou_threshold, 
                imgsz=request.imgsz
            )
            
            logger.info(f"[{request_id}] Memory after icon processing: {check_memory_usage()}")
            
            # 清理不再需要的中间变量
            del image_input, ocr_bbox, draw_bbox_config, text
            
            # 格式化解析结果
            parsed_elements = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])
            
            # 清理不再需要的变量
            del label_coordinates, parsed_content_list
        
        # 准备返回结果
        response = ParseResponse(
            image=dino_labled_img,
            parsed_elements=parsed_elements
        )
        
        # 清理大型变量
        del dino_labled_img, parsed_elements
        
        logger.info(f"[{request_id}] Memory before returning response: {check_memory_usage()}")
        logger.info(f"[{request_id}] Processing completed successfully")
        
        # 返回结果
        return response
    
    except HTTPException as e:
        # 记录错误信息
        logger.error(f"[{request_id}] HTTPException: {e.detail} (code: {e.status_code})")
        # 重新抛出 HTTPException，保持其原始状态码和详细信息
        raise
    
    except Exception as e:
        # 记录错误信息
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        # 处理任何其他未捕获的异常
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
    
    finally:
        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 触发 Python 垃圾回收
        import gc
        gc.collect()
        
        logger.info(f"[{request_id}] Memory after cleanup: {check_memory_usage()}")
        
        # 释放锁
        if process_lock.locked():
            process_lock.release()

# 配置异常处理
@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Unexpected error: {str(exc)}"}
    )

# 启动服务器（如果直接运行此文件）
if __name__ == "__main__":
    # 启动应用，模型已经在应用创建前加载完成
    logger.info("Starting OmniParser API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 