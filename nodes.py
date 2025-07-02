import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import os
import time

# --- Extracted from client_example.py ---
class RequestParams:
    def __init__(self, optimizer_lr_stroke, optimizer_color_lr, num_steps_stroke,
                 num_strokes, width_scale, length_scale, print_freq,
                 scale_by_y, init):
        self.optimizer_lr_stroke = optimizer_lr_stroke
        self.optimizer_color_lr = optimizer_color_lr
        self.num_steps_stroke = num_steps_stroke
        self.num_strokes = num_strokes
        self.width_scale = width_scale
        self.length_scale = length_scale
        self.print_freq = print_freq
        self.scale_by_y = scale_by_y
        self.init = init

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
            "init": self.init
        }

class StrokeOptimRequest:
    def __init__(self, content_img, params, use_comet=False):
        self.content_img = content_img
        self.params = params
        self.use_comet = use_comet

    def submit(self, base_url, api_key):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": {
                "content_img": self.content_img,
                "params": self.params.to_dict(),
                "use_comet": self.use_comet
            }
        }
        response = requests.post(f"{base_url}/run", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["id"]

    def poll(self, base_url, api_key, job_id):
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        while True:
            response = requests.get(f"{base_url}/status/{job_id}", headers=headers)
            response.raise_for_status()
            status = response.json()
            
            if status["status"] == "COMPLETED":
                # Assuming the final result is in the last element of the output list
                final_output = status["output"][-1]
                svg_str = final_output["svg"]
                img_base64 = final_output["img"]
                
                # Decode base64 image to PIL Image
                img_bytes = base64.b64decode(img_base64)
                img = Image.open(io.BytesIO(img_bytes))
                
                yield {"step": "final", "is_final": True, "svg": svg_str, "img": img, "loss": final_output.get("loss")}
                break
            elif status["status"] == "FAILED":
                raise RuntimeError(f"Job failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "IN_PROGRESS" and status.get("output"):
                # Yield intermediate results if available
                for output_item in status["output"]:
                    if "svg" in output_item and "img" in output_item:
                        svg_str = output_item["svg"]
                        img_base64 = output_item["img"]
                        
                        img_bytes = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        yield {"step": output_item.get("step"), "is_final": False, "svg": svg_str, "img": img, "loss": output_item.get("loss")}
            
            time.sleep(5) # Poll every 5 seconds

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
                "optimizer_color_lr": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "num_steps_stroke": ("INT", {"default": 5001, "min": 1, "max": 100000, "step": 100}),
                "num_strokes": ("INT", {"default": 250, "min": 1, "max": 1000}),
                "width_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "length_scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "print_freq": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "scale_by_y": ("BOOLEAN", {"default": True}),
                "init": (["random", "content"], {"default": "random"}),
            },
            "optional": {
                "use_comet": ("BOOLEAN", {"default": False}), # Added from client_example.py
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
                                        scale_by_y, init, use_comet=False):
        
        if not api_key:
            raise ValueError("API Key is required.")
        if not base_url:
            raise ValueError("Base URL is required.")

        # 1. Convert ComfyUI IMAGE (torch.Tensor) to base64
        # Assuming image is a batch of 1, and in [0,1] float32
        i = 255. * image.cpu().numpy().squeeze() # Remove batch dim, scale to 0-255
        img_pil = Image.fromarray(np.uint8(i)) # Convert to PIL Image
        
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG") # Save as PNG to buffer
        content_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 2. Construct RequestParams
        params = RequestParams(
            optimizer_lr_stroke=optimizer_lr_stroke,
            optimizer_color_lr=optimizer_color_lr,
            num_steps_stroke=num_steps_stroke,
            num_strokes=num_strokes,
            width_scale=width_scale,
            length_scale=length_scale,
            print_freq=print_freq,
            scale_by_y=scale_by_y,
            init=init
        )

        # 3. Create StrokeOptimRequest and submit job
        request_obj = StrokeOptimRequest(
            content_img=content_img_base64,
            params=params,
            use_comet=use_comet
        )

        try:
            job_id = request_obj.submit(base_url, api_key)
            print(f"Submitted job with ID: {job_id}")

            # 4. Poll for job completion
            final_svg_str = None
            final_img_pil = None
            for result in request_obj.poll(base_url, api_key, job_id):
                if result["is_final"]:
                    final_svg_str = result["svg"]
                    final_img_pil = result["img"]
                    print(f"Job {job_id} completed. Final loss: {result.get('loss')}")
                else:
                    print(f"Job {job_id} in progress, step: {result.get('step')}, loss: {result.get('loss')}")
            
            if final_svg_str is None or final_img_pil is None:
                raise RuntimeError("Failed to retrieve final results from the API.")

            # 5. Convert final PIL Image to ComfyUI IMAGE (torch.Tensor)
            img_np = np.array(final_img_pil).astype(np.float32) / 255.0
            # Add batch dimension and ensure 3 channels (even if grayscale)
            if len(img_np.shape) == 2: # Grayscale
                img_np = np.expand_dims(img_np, axis=-1) # Add channel dim
            if img_np.shape[2] == 4: # RGBA to RGB
                img_np = img_np[..., :3]
            img_tensor = torch.from_numpy(img_np)[None,] # Add batch dim

            return (final_svg_str, img_tensor,)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API Request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "MatrBrushstrokes": MatrBrushstrokesNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MatrBrushstrokes": "Matr Brushstrokes API"
}