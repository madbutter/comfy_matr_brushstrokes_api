import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import os
import time
import gzip # Keep import for robustness, but won't be used for SVG decompression
import json # Added for potential future use, good practice for API responses

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

    def poll(self, base_url, api_key, job_id, verbose_output=False):
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        while True:
            response = requests.get(f"{base_url}/status/{job_id}", headers=headers)
            response.raise_for_status()
            status = response.json()
            
            if verbose_output:
                print(f"API Status for job {job_id}: {json.dumps(status, indent=2)}") # Pretty print JSON
            
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
                "optimizer_lr_stroke": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001}), # User's default
                "optimizer_color_lr": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}), # User's default
                "num_steps_stroke": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}), # User's default
                "num_strokes": ("INT", {"default": 3500, "min": 1, "max": 10000, "step": 1}), # User's default
                "width_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}), # User's default
                "length_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}), # User's default
                "print_freq": ("INT", {"default": 100, "min": 1, "max": 1000}), # User's default
                "scale_by_y": ("BOOLEAN", {"default": False}),
                "init": (["random", "slic"], {"default": "slic"}), # User's default
                "verbose_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "use_comet": ("BOOLEAN", {"default": False}),
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
                                        scale_by_y, init, verbose_output, use_comet=False):
        
        if not api_key:
            raise ValueError("API Key is required.")
        if not base_url:
            raise ValueError("Base URL is required.")

        i = 255. * image.cpu().numpy().squeeze()
        img_pil = Image.fromarray(np.uint8(i))
        
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        content_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

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

        request_obj = StrokeOptimRequest(
            content_img=content_img_base64,
            params=params,
            use_comet=use_comet
        )

        try:
            job_id = request_obj.submit(base_url, api_key)
            print(f"Submitted job with ID: {job_id}")

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

            img_np = np.array(final_img_pil).astype(np.float32) / 255.0
            if len(img_np.shape) == 2:
                img_np = np.expand_dims(img_np, axis=-1)
            if img_np.shape[2] == 4:
                img_np = img_np[..., :3]
            img_tensor = torch.from_numpy(img_np)[None,]

            return (final_svg_str, img_tensor,)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API Request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")

# --- New Helper Node to Decode and Save SVG ---
class SaveDecodedSVGNode:
    def __init__(self):
        # Get ComfyUI's base directory (assuming node is in custom_nodes/your_node_folder)
        comfy_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
        self.output_dir = os.path.join(comfy_base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_encoded_svg_string": ("STRING",), # Changed to standard STRING input
                "filename_prefix": ("STRING", {"default": "ComfyUI_MatrBrushstrokes_SVG"}),
            }
        }

    RETURN_TYPES = () # This is an output node, no direct output connections
    FUNCTION = "save_svg"
    OUTPUT_NODE = True # Mark as an output node
    CATEGORY = "Image/MatrBrushstrokes"

    def save_svg(self, base64_encoded_svg_string, filename_prefix):
        try:
            # 1. Base64 decode
            decoded_bytes = base64.b64decode(base64_encoded_svg_string)
            
            svg_xml_string = None
            
            # Try Gzip decompression first (as per handler.py, though it might not be gzipped in practice)
            try:
                decompressed_bytes = gzip.decompress(decoded_bytes)
                svg_xml_string = decompressed_bytes.decode('utf-8')
                print("Attempted Gzip decompression: SUCCESS")
            except gzip.BadGzipFile:
                # If not gzipped, assume it's plain base64 encoded SVG XML
                print("Attempted Gzip decompression: FAILED. Assuming plain Base64 encoded SVG.")
                svg_xml_string = decoded_bytes.decode('utf-8')
            
            if svg_xml_string is None:
                raise RuntimeError("Failed to decode SVG string.")

            # 4. Save to file
            # Generate a unique filename
            timestamp = int(time.time())
            filename = f"{filename_prefix}_{timestamp}.svg"
            file_path = os.path.join(self.output_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(svg_xml_string)
            
            print(f"SVG successfully decoded and saved to: {file_path}")
            
        except base64.binascii.Error as e:
            raise RuntimeError(f"SVG Decoding Error (Base64): {e}. Input string might be invalid.")
        except Exception as e: # Catch any other unexpected errors
            raise RuntimeError(f"An unexpected error occurred while saving SVG: {e}")
        
        return {} # Output nodes typically return an empty dictionary

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "MatrBrushstrokes": MatrBrushstrokesNode,
    "SaveDecodedSVG": SaveDecodedSVGNode, # Added new node
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MatrBrushstrokes": "Matr Brushstrokes API",
    "SaveDecodedSVG": "Save Decoded SVG", # Added new node display name
}