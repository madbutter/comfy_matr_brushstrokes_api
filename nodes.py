import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import os
import time
import gzip
import json

# --- Extracted from client_example.py and request.py analysis ---
class RequestParams:
    def __init__(self, optimizer_lr_stroke, optimizer_color_lr, num_steps_stroke,
                 num_strokes, width_scale, length_scale, print_freq,
                 scale_by_y, init, mask_alpha=False):
        self.optimizer_lr_stroke = optimizer_lr_stroke
        self.optimizer_color_lr = optimizer_color_lr
        self.num_steps_stroke = num_steps_stroke
        self.num_strokes = num_strokes
        self.width_scale = width_scale
        self.length_scale = length_scale
        self.print_freq = print_freq
        self.scale_by_y = scale_by_y
        self.init = init
        self.mask_alpha = mask_alpha  # New parameter for mask support

    def to_dict(self):
        return {
            "optimizer_lr_stroke": self.optimizer_lr_stroke,
            "optimizer_color_lr": self.optimizer_color_lr,
            "num_steps_stroke": self.num_steps_stroke,
            "num_strokes": self.num_strokes,
            "width_scale": self.width_scale,
            "length_scale": self.length_scale,
            "print_freq": self.print_freq,
            "scale_by_y": self.scale_by_y,
            "init": self.init,
            "mask_alpha": self.mask_alpha,  # Include mask_alpha in API payload
            # These parameters are now omitted from the node's inputs,
            # but are included here with their default values as per request.py
            "img_size": 512,
            "samples_per_curve": 10,
            "brushes_per_pixel": 20,
            "canvas_color": "black",
            "style_weight_stroke": 0.0,
            "content_weight_stroke": 10.0,
            "tv_weight_stroke": 0.008,
            "tv_loss_k": 1,
            "curv_weight": 2.0,
            "optimize_width": False
        }

class StrokeOptimRequest:
    def __init__(self, content_img, params, mask_img=None):
        self.content_img = content_img
        self.params = params
        self.mask_img = mask_img  # New mask image parameter
        # use_comet, style_img, experiment_name, experiment_tags are omitted, API will use its defaults
        # or they are not needed for basic functionality

    def submit(self, base_url, api_key):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": {
                "content_img": self.content_img,
                "params": self.params.to_dict(),
                # use_comet is omitted, API will use its default
            }
        }
        
        # Add mask_img to payload if provided
        if self.mask_img is not None:
            payload["input"]["mask_img"] = self.mask_img
        
        # style_img, experiment_name, experiment_tags are omitted from payload

        response = requests.post(f"{base_url}/run", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["id"]

    def poll(self, base_url, api_key, job_id, verbose_output=False):
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        while True:
            response = requests.get(f"{base_url}/status/{job_id}", headers=headers)
            response.raise_for_status()
            status = response.json()
            
            if verbose_output:
                print(f"API Status for job {job_id}: {json.dumps(status, indent=2)}")
            
            if status["status"] == "COMPLETED":
                if not status.get("output"):
                    raise RuntimeError(f"Job {job_id} completed but no output found.")
                
                final_output = status["output"][-1]
                
                if "svg" not in final_output:
                    raise RuntimeError(f"Job {job_id} completed but 'svg' key missing in final output: {final_output}")
                if "image" not in final_output:
                    raise RuntimeError(f"Job {job_id} completed but 'image' key missing in final output: {final_output}")

                svg_str = final_output["svg"]
                img_base64 = final_output["image"]
                
                img_bytes = base64.b64decode(img_base64)
                img = Image.open(io.BytesIO(img_bytes))
                
                yield {"step": "final", "is_final": True, "svg": svg_str, "image": img, "loss": final_output.get("loss")}
                break
            elif status["status"] == "FAILED":
                raise RuntimeError(f"Job failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "IN_PROGRESS" and status.get("output"):
                for output_item in status["output"]:
                    if "svg" in output_item and "image" in output_item:
                        svg_str = output_item["svg"]
                        img_base64 = output_item["image"]
                        
                        img_bytes = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        yield {"step": output_item.get("step"), "is_final": False, "svg": svg_str, "image": img, "loss": output_item.get("loss")}
            
            time.sleep(5)

# --- Helper function to convert ComfyUI image to mask base64 ---
def image_to_mask_base64(image_tensor):
    """
    Convert ComfyUI image tensor to base64 encoded PNG mask image.
    Accepts any image type (1-bit, 8-bit grayscale, 24-bit RGB) and converts to grayscale mask.
    White areas (255) = mask area, black areas (0) = background.
    """
    # Convert from tensor to numpy - image_tensor is typically [batch, height, width, channels]
    img_np = image_tensor.cpu().numpy().squeeze()  # Remove batch dimension
    
    # Handle different input formats
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # RGB image
        # Convert RGB to grayscale using standard luminance formula
        img_np = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    elif len(img_np.shape) == 3 and img_np.shape[2] == 4:  # RGBA image
        # Convert RGB to grayscale, ignore alpha channel
        img_np = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    elif len(img_np.shape) == 3 and img_np.shape[2] == 1:  # Single channel with extra dim
        img_np = img_np.squeeze()
    # If len(img_np.shape) == 2, it's already grayscale
    
    # Ensure values are in 0-1 range then convert to 0-255
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)
    
    # Convert to PIL Image (grayscale)
    mask_pil = Image.fromarray(img_np, mode='L')
    
    # Convert to base64
    buffered = io.BytesIO()
    mask_pil.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return mask_base64

# --- ComfyUI Node Definition ---
class MatrBrushstrokesNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": os.environ.get("MATR_RUNPOD_API_KEY", "")}),
                "base_url": ("STRING", {"multiline": False, "default": "https://api.runpod.ai/v2/vt0tvhzcqug7zu"}),
                "optimizer_lr_stroke": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001}),
                "optimizer_color_lr": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "num_steps_stroke": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "num_strokes": ("INT", {"default": 3500, "min": 1, "max": 6000}),
                "width_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "length_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "print_freq": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "scale_by_y": ("BOOLEAN", {"default": False}),
                "init": (["random", "slic"], {"default": "slic"}),
                "verbose_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("IMAGE",),  # New optional mask input - accepts any image type
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("SVG_Output", "Image_Output",)
    FUNCTION = "process_image_with_brushstrokes"
    CATEGORY = "Image/MatrBrushstrokes"

    def process_image_with_brushstrokes(self, image, api_key, base_url,
                                        optimizer_lr_stroke, optimizer_color_lr,
                                        num_steps_stroke, num_strokes,
                                        width_scale, length_scale, print_freq,
                                        scale_by_y, init, verbose_output, mask=None):
        
        if not api_key:
            raise ValueError("API Key is required.")
        if not base_url:
            raise ValueError("Base URL is required.")

        # Convert main image to base64
        i = 255. * image.cpu().numpy().squeeze()
        img_pil = Image.fromarray(np.uint8(i))
        
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        content_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Handle mask input
        mask_img_base64 = None
        mask_alpha = False
        
        if mask is not None:
            print("Mask detected - enabling transparency mode")
            mask_alpha = True
            mask_img_base64 = image_to_mask_base64(mask)

        # Create parameters with mask_alpha setting
        params = RequestParams(
            optimizer_lr_stroke=optimizer_lr_stroke,
            optimizer_color_lr=optimizer_color_lr,
            num_steps_stroke=num_steps_stroke,
            num_strokes=num_strokes,
            width_scale=width_scale,
            length_scale=length_scale,
            print_freq=print_freq,
            scale_by_y=scale_by_y,
            init=init,
            mask_alpha=mask_alpha  # Automatically set based on mask presence
        )

        # Create request with optional mask
        request_obj = StrokeOptimRequest(
            content_img=content_img_base64,
            params=params,
            mask_img=mask_img_base64  # Will be None if no mask provided
        )

        try:
            job_id = request_obj.submit(base_url, api_key)
            print(f"Submitted job with ID: {job_id}")
            if mask_alpha:
                print("Job submitted with mask - output will have transparency")

            final_svg_str = None
            final_img_pil = None
            for result in request_obj.poll(base_url, api_key, job_id, verbose_output):
                if result["is_final"]:
                    final_svg_str = result["svg"]
                    final_img_pil = result["image"]
                    print(f"Job {job_id} completed. Final loss: {result.get('loss')}")
                else:
                    print(f"Job {job_id} in progress, step: {result.get('step')}, loss: {result.get('loss')}")
            
            if final_svg_str is None or final_img_pil is None:
                raise RuntimeError("Failed to retrieve final results from the API.")

            # Convert PIL image to tensor
            img_np = np.array(final_img_pil).astype(np.float32) / 255.0
            if len(img_np.shape) == 2: # Grayscale
                img_np = np.expand_dims(img_np, axis=-1) # Add channel dim
            
            # Handle both RGB and RGBA outputs (RGBA when mask_alpha=True)
            if img_np.shape[2] == 4: # RGBA
                if mask_alpha:
                    print("Received RGBA output with transparency - keeping alpha channel")
                    # Keep RGBA for transparent output
                else:
                    # If we unexpectedly got RGBA without mask, convert to RGB
                    print("Received unexpected RGBA output - converting to RGB")
                    img_np = img_np[..., :3]
            elif img_np.shape[2] == 3: # RGB
                if mask_alpha:
                    print("Expected RGBA with transparency but got RGB - this may indicate API processing issue")
                # Keep as RGB
            
            img_tensor = torch.from_numpy(img_np)[None,] # Add batch dim

            return (final_svg_str, img_tensor,)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API Request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")

# --- New Helper Node to Decode and Save SVG ---
class SaveDecodedSVGNode:
    def __init__(self):
        comfy_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
        self.output_dir = os.path.join(comfy_base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_encoded_svg_string": ("STRING",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_MatrBrushstrokes_SVG"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_svg"
    OUTPUT_NODE = True
    CATEGORY = "Image/MatrBrushstrokes"

    def save_svg(self, base64_encoded_svg_string, filename_prefix):
        try:
            decoded_bytes = base64.b64decode(base64_encoded_svg_string)
            
            svg_xml_string = None
            
            try:
                decompressed_bytes = gzip.decompress(decoded_bytes)
                svg_xml_string = decompressed_bytes.decode('utf-8')
                print("Attempted Gzip decompression: SUCCESS")
            except gzip.BadGzipFile:
                print("Attempted Gzip decompression: FAILED. Assuming plain Base64 encoded SVG.")
                svg_xml_string = decoded_bytes.decode('utf-8')
            
            if svg_xml_string is None:
                raise RuntimeError("Failed to decode SVG string.")

            timestamp = int(time.time())
            filename = f"{filename_prefix}_{timestamp}.svg"
            file_path = os.path.join(self.output_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(svg_xml_string)
            
            print(f"SVG successfully decoded and saved to: {file_path}")
            
        except base64.binascii.Error as e:
            raise RuntimeError(f"SVG Decoding Error (Base64): {e}. Input string might be invalid.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while saving SVG: {e}")
        
        return {}

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "MatrBrushstrokes": MatrBrushstrokesNode,
    "SaveDecodedSVG": SaveDecodedSVGNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MatrBrushstrokes": "Matr Brushstrokes API",
    "SaveDecodedSVG": "Matr Save Decoded SVG",
}