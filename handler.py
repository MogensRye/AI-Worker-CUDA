"""
Production RunPod Serverless AI Worker
GPU-accelerated video background replacement using SAM2 Heavy + MatAnyone
Optimized for speed, cost, and reliability on RunPod infrastructure
"""

import os
import sys
import time
import tempfile
import subprocess
import glob
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

import runpod
import requests
import torch
import numpy as np
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instances
SAM2_PREDICTOR = None
MATANYONE_PROCESSOR = None
MODEL_READY = False

# Performance settings
KEYFRAME_STRIDE = 30  # Process every 30th frame for masks
MAX_SIZE = 1080       # Max resolution for processing
USE_FP16 = True       # Mixed precision for speed
NVENC_AVAILABLE = None

def check_nvenc_support() -> bool:
    """Check if NVENC hardware encoding is available"""
    global NVENC_AVAILABLE
    if NVENC_AVAILABLE is not None:
        return NVENC_AVAILABLE
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'], 
            capture_output=True, text=True, timeout=10
        )
        NVENC_AVAILABLE = 'h264_nvenc' in result.stdout
        logger.info(f"🎬 NVENC support: {'✅ Available' if NVENC_AVAILABLE else '❌ Not available'}")
        return NVENC_AVAILABLE
    except Exception as e:
        logger.warning(f"⚠️ Could not check NVENC support: {e}")
        NVENC_AVAILABLE = False
        return False

def cold_start() -> bool:
    """Initialize all AI models on GPU with optimizations"""
    global MODEL_READY, SAM2_PREDICTOR, MATANYONE_PROCESSOR
    
    if MODEL_READY:
        return True
    
    start_time = time.time()
    logger.info("🚀 Starting cold start - loading AI models...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available - GPU required for this worker")
        return False
    
    device = torch.device("cuda")
    logger.info(f"🎯 Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Load SAM2 Heavy
    try:
        logger.info("📥 Loading SAM2 Heavy model...")
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        SAM2_PREDICTOR = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large",
            device=device
        )
        
        # Enable mixed precision if supported
        if USE_FP16 and hasattr(SAM2_PREDICTOR.model, 'half'):
            SAM2_PREDICTOR.model.half()
            logger.info("🔥 SAM2 using FP16 mixed precision")
        
        logger.info("✅ SAM2 Heavy loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load SAM2: {e}")
        return False
    
    # Load MatAnyone
    try:
        logger.info("📥 Loading MatAnyone processor...")
        from matanyone import InferenceCore
        
        MATANYONE_PROCESSOR = InferenceCore("PeiqingYang/MatAnyone")
        logger.info("✅ MatAnyone loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load MatAnyone: {e}")
        return False
    
    # Check NVENC support
    check_nvenc_support()
    
    MODEL_READY = True
    load_time = time.time() - start_time
    logger.info(f"🎉 Cold start completed in {load_time:.2f}s")
    
    return True

def download_file(url: str, dest_path: str, timeout: int = 120) -> bool:
    """Download file with progress tracking and error handling"""
    try:
        logger.info(f"📥 Downloading: {url}")
        
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"📥 Progress: {progress:.1f}%")
        
        logger.info(f"✅ Downloaded: {dest_path} ({downloaded / 1024 / 1024:.1f}MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def extract_keyframes(video_path: str, output_dir: str, stride: int = KEYFRAME_STRIDE) -> List[str]:
    """Extract keyframes from video for SAM2 processing"""
    ensure_dir(output_dir)
    
    # Use ffmpeg to extract keyframes
    frame_pattern = os.path.join(output_dir, "frame_%06d.png")
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'select=not(mod(n\\,{stride}))',
        '-vsync', 'vfr',
        '-q:v', '2',  # High quality
        frame_pattern,
        '-y'  # Overwrite
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"❌ FFmpeg keyframe extraction failed: {result.stderr}")
            return []
        
        # Get extracted frames
        frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
        logger.info(f"🎬 Extracted {len(frames)} keyframes (stride={stride})")
        return frames
        
    except Exception as e:
        logger.error(f"❌ Keyframe extraction failed: {e}")
        return []

def generate_sam2_masks(frames: List[str], masks_dir: str) -> bool:
    """Generate SAM2 masks for keyframes"""
    if not SAM2_PREDICTOR:
        logger.error("❌ SAM2 not loaded")
        return False
    
    ensure_dir(masks_dir)
    
    for i, frame_path in enumerate(frames):
        try:
            logger.info(f"🎯 Processing frame {i+1}/{len(frames)}: {os.path.basename(frame_path)}")
            
            # Load image
            image = Image.open(frame_path).convert("RGB")
            image_np = np.array(image)
            
            # Resize if too large
            h, w = image_np.shape[:2]
            if max(h, w) > MAX_SIZE:
                scale = MAX_SIZE / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                image_np = np.array(image)
                logger.info(f"📏 Resized from {w}x{h} to {new_w}x{new_h}")
            
            # SAM2 segmentation
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                masks = SAM2_PREDICTOR.predict_everything(image, verbose=False)
            
            # Combine masks (assume person is largest connected component)
            if masks:
                combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                for mask_result in masks:
                    combined_mask = np.logical_or(combined_mask, mask_result.mask)
                
                # Convert to binary mask
                mask_final = combined_mask.astype(np.uint8) * 255
                
                # Morphological operations to clean up mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
                mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
                
            else:
                logger.warning(f"⚠️ No masks found for {frame_path}")
                mask_final = np.zeros(image_np.shape[:2], dtype=np.uint8)
            
            # Save mask with same naming pattern as frame
            frame_name = os.path.basename(frame_path)
            mask_name = frame_name.replace("frame_", "mask_")
            mask_path = os.path.join(masks_dir, mask_name)
            
            cv2.imwrite(mask_path, mask_final)
            logger.info(f"💾 Saved mask: {mask_path}")
            
        except Exception as e:
            logger.error(f"❌ SAM2 processing failed for {frame_path}: {e}")
            return False
    
    return True

def process_with_matanyone(video_path: str, background_path: str, masks_dir: str, output_dir: str) -> Optional[str]:
    """Process video with MatAnyone using SAM2 masks"""
    if not MATANYONE_PROCESSOR:
        logger.error("❌ MatAnyone not loaded")
        return None
    
    try:
        logger.info("🎨 Starting MatAnyone background replacement...")
        
        # Process with MatAnyone
        result = MATANYONE_PROCESSOR.process_video(
            input_path=video_path,
            mask_path=masks_dir,  # Directory with masks
            background_path=background_path,
            output_path=output_dir,
            max_size=MAX_SIZE,
            save_frames=False
        )
        
        if result and len(result) >= 2:
            foreground_video, alpha_video = result[:2]
            logger.info(f"✅ MatAnyone processing complete")
            logger.info(f"📹 Foreground: {foreground_video}")
            logger.info(f"🎭 Alpha: {alpha_video}")
            return foreground_video, alpha_video
        else:
            logger.error("❌ MatAnyone returned invalid result")
            return None
            
    except Exception as e:
        logger.error(f"❌ MatAnyone processing failed: {e}")
        return None

def compose_final_video(foreground_path: str, alpha_path: str, background_path: str, 
                       original_video: str, output_path: str) -> bool:
    """Compose final video with background replacement and audio"""
    try:
        logger.info("🎬 Composing final video with FFmpeg...")
        
        # Choose encoder based on NVENC availability
        if check_nvenc_support():
            video_codec = 'h264_nvenc'
            codec_params = ['-preset', 'fast', '-cq', '23']
            logger.info("🚀 Using NVENC hardware encoding")
        else:
            video_codec = 'libx264'
            codec_params = ['-preset', 'medium', '-crf', '23']
            logger.info("🐌 Using CPU encoding (libx264)")
        
        # FFmpeg command for background replacement
        cmd = [
            'ffmpeg',
            '-i', foreground_path,      # Foreground video
            '-i', alpha_path,           # Alpha matte
            '-i', background_path,      # Background image
            '-i', original_video,       # Original video (for audio)
            '-filter_complex', 
            '[2:v]scale=iw:ih[bg];'     # Scale background
            '[0:v][1:v][bg]alphamerge=alpha_mode=straight[composed]',  # Compose
            '-map', '[composed]',       # Use composed video
            '-map', '3:a?',            # Copy audio from original (if exists)
            '-c:v', video_codec,        # Video codec
            *codec_params,              # Codec parameters
            '-c:a', 'aac',             # Audio codec
            '-shortest',                # Match shortest stream
            '-y',                       # Overwrite output
            output_path
        ]
        
        logger.info(f"🎬 Running FFmpeg composition...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ Video composition successful: {output_path}")
            return True
        else:
            logger.error(f"❌ FFmpeg composition failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Video composition failed: {e}")
        return False

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def cleanup_temp_files(*paths):
    """Clean up temporary files and directories"""
    for path in paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up {path}: {e}")

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler for video background replacement
    
    Expected input:
    {
      "input": {
        "video_url": "https://.../input.mp4",
        "background_url": "https://.../background.jpg",
        "output_name": "result.mp4"  # optional
      }
    }
    """
    start_time = time.time()
    
    # Initialize models
    if not cold_start():
        return {
            "error": "Failed to initialize AI models",
            "status": "error"
        }
    
    # Parse input
    data = event.get("input", {})
    video_url = data.get("video_url")
    background_url = data.get("background_url")
    output_name = data.get("output_name", "result.mp4")
    
    if not video_url or not background_url:
        return {
            "error": "Both video_url and background_url are required",
            "status": "error"
        }
    
    # Create workspace
    workspace = f"/tmp/workspace/{int(time.time())}"
    ensure_dir(workspace)
    
    try:
        logger.info(f"🎬 Starting video background replacement job")
        logger.info(f"📹 Video: {video_url}")
        logger.info(f"🖼️ Background: {background_url}")
        
        # Download input files
        video_path = os.path.join(workspace, "input.mp4")
        background_path = os.path.join(workspace, "background.jpg")
        
        if not download_file(video_url, video_path):
            return {"error": "Failed to download video", "status": "error"}
        
        if not download_file(background_url, background_path):
            return {"error": "Failed to download background", "status": "error"}
        
        # Create working directories
        frames_dir = os.path.join(workspace, "frames")
        masks_dir = os.path.join(workspace, "masks")
        matanyone_dir = os.path.join(workspace, "matanyone")
        
        # Step 1: Extract keyframes
        logger.info("🎬 Step 1: Extracting keyframes...")
        keyframes = extract_keyframes(video_path, frames_dir, KEYFRAME_STRIDE)
        if not keyframes:
            return {"error": "Failed to extract keyframes", "status": "error"}
        
        # Step 2: Generate SAM2 masks
        logger.info("🎯 Step 2: Generating SAM2 masks...")
        if not generate_sam2_masks(keyframes, masks_dir):
            return {"error": "Failed to generate SAM2 masks", "status": "error"}
        
        # Step 3: Process with MatAnyone
        logger.info("🎨 Step 3: Processing with MatAnyone...")
        matanyone_result = process_with_matanyone(video_path, background_path, masks_dir, matanyone_dir)
        if not matanyone_result:
            return {"error": "MatAnyone processing failed", "status": "error"}
        
        foreground_video, alpha_video = matanyone_result
        
        # Step 4: Compose final video
        logger.info("🎬 Step 4: Composing final video...")
        output_path = os.path.join(workspace, output_name)
        if not compose_final_video(foreground_video, alpha_video, background_path, video_path, output_path):
            return {"error": "Video composition failed", "status": "error"}
        
        # Get file info
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        processing_time = time.time() - start_time
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"⏱️ Total time: {processing_time:.2f}s")
        logger.info(f"📁 Output size: {file_size:.1f}MB")
        
        return {
            "status": "success",
            "output_path": output_path,
            "file_size_mb": round(file_size, 2),
            "processing_time_seconds": round(processing_time, 2),
            "models_used": {
                "sam2": "facebook/sam2-hiera-large",
                "matanyone": "PeiqingYang/MatAnyone"
            },
            "settings": {
                "keyframe_stride": KEYFRAME_STRIDE,
                "max_size": MAX_SIZE,
                "use_fp16": USE_FP16,
                "nvenc_used": check_nvenc_support()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Handler failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
    
    finally:
        # Cleanup temporary files
        cleanup_temp_files(workspace)

if __name__ == "__main__":
    # Start RunPod serverless worker
    logger.info("🚀 Starting RunPod Serverless AI Worker...")
    runpod.serverless.start({"handler": handler})
