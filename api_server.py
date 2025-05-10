import io
import base64
import threading
import time
from typing import Optional, Dict, List, Any, Union
import asyncio
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行的代码
    print("INFO:     Initiating model loading in background...")
    # 在单独的线程中运行同步的 load_models 函数，
    # 这样它就不会阻塞应用程序的启动。
    asyncio.create_task(asyncio.to_thread(load_models))
    yield
    # 应用关闭时执行的代码 (如果需要)
    print("INFO:     Application shutdown.")

# 初始化 FastAPI 应用
app = FastAPI(
    title="OmniParser API",
    description="API for parsing GUI screens into structured elements",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
models_loaded = False
model_loading = False
yolo_model = None
caption_model_processor = None
process_lock = threading.Lock()

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

# 辅助函数：将 base64 字符串转换为 PIL 图像
def base64_to_pil(image_base64: str) -> Image.Image:
    """
    将 base64 编码的图像字符串转换为 PIL Image 对象
    """
    # 改进的错误处理 - 将所有可能的错误转为 400 Bad Request
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
        except Exception as e:
            raise ValueError(f"Base64 decoding error: {str(e)}")
        
        # 尝试打开图像
        try:
            image = Image.open(io.BytesIO(image_data))
            # 验证是否是有效图像
            image.verify()  # 验证图像数据
            image = Image.open(io.BytesIO(image_data))  # 需要重新打开，因为 verify() 会消耗文件指针
            return image
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
            
    except Exception as e:
        # 捕获所有异常，并以 400 Bad Request 抛出
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )

# 加载模型函数
def load_models():
    global yolo_model, caption_model_processor, models_loaded, model_loading
    
    if models_loaded or model_loading:
        return
    
    model_loading = True
    try:
        # 加载 YOLO 模型
        yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
        
        # 加载 Florence-2 模型
        caption_model_processor = get_caption_model_processor(
            model_name="florence2", 
            model_name_or_path="weights/icon_caption_florence"
        )
        
        models_loaded = True
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        models_loaded = False
    finally:
        model_loading = False

# 健康检查端点
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    if not models_loaded:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        status_message = "Service unavailable - Models not loaded"
    else:
        status_code = status.HTTP_200_OK
        status_message = "Service available"
    
    response = HealthResponse(
        status=status_message,
        models_loaded=models_loaded,
        timestamp=time.time()
    )
    
    # 更新 dict() 为 model_dump() - 修复 Pydantic V2 弃用警告
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump()
    )

# 图像解析端点
@app.post("/api/parse", response_model=ParseResponse)
async def parse_image(request: ParseRequest):
    # 检查模型是否已加载
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading, please try again later"
        )
    
    # 尝试获取处理锁
    if not process_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is currently processing another request, please try again later"
        )
    
    try:
        # 将 base64 图像转换为 PIL Image
        # 现在 base64_to_pil 函数会直接抛出 HTTPException，状态码为 400
        image_input = base64_to_pil(request.image)
        
        # 准备图像处理参数
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        # 执行 OCR
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_input, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=request.use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # 获取标记后的图像和解析结果
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
        
        # 格式化解析结果
        parsed_elements = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])
        
        # 返回结果
        return ParseResponse(
            image=dino_labled_img,
            parsed_elements=parsed_elements
        )
    
    except HTTPException:
        # 重新抛出 HTTPException，保持其原始状态码和详细信息
        raise
    
    except Exception as e:
        # 处理任何其他未捕获的异常
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
    
    finally:
        # 释放锁
        if process_lock.locked():
            process_lock.release()

# 配置异常处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Unexpected error: {str(exc)}"}
    )

# 启动服务器（如果直接运行此文件）
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        workers=1  # 确保只有一个工作进程以避免模型重复加载
    ) 