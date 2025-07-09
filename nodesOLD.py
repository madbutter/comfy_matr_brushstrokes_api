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

# --- Extracted from client_example.py and handler.py analysis ---
class RequestParams:
    def __init__(self, optimizer_lr_stroke, optimizer_color_lr, num_steps_stroke,
                 num_strokes, width_scale, length_scale, print_freq,
                 scale_by_y, init,
                 # Added parameters from handler.py analysis
                 img_size, samples_per_curve, brushes_per_pixel, canvas_color,
                 style_weight_stroke, content_weight_stroke, tv_weight_stroke,
                 tv_loss_k, curv_weight, optimize_width):
        self.optimizer_lr_stroke = optimizer_lr_stroke
        self.optimizer_color_lr = optimizer_color_lr
        self.num_steps_stroke = num_steps_stroke
        self.num_strokes = num_strokes
        self.width_scale = width_scale
        self.length_scale = length_scale
        self.print_freq = print_freq
        self.scale_by_y = scale_by_y
        self.init = init
        # Added parameters
        self.img_size = img_size
        self.samples_per_curve = samples_per_curve
        self.brushes_per_pixel = brushes_per_pixel
        self.canvas_color = canvas_color
        self.style_weight_stroke = style_weight_stroke
        self.content_weight_stroke = content_weight_stroke
        self.tv_weight_stroke = tv_weight_stroke
        self.tv_loss_k = tv_loss_k
        self.curv_weight = curv_weight
        self.optimize_width = optimize_width

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
            # Added parameters
            "img_size": self.img_size,
            "samples_per_curve": self.samples_per_curve,
            "brushes_per_pixel": self.brushes_per_pixel,
            "canvas_color": self.canvas_color,
            "style_weight_stroke": self.style_weight_stroke,
            "content_weight_stroke": self.content_weight_stroke,
            "tv_weight_stroke": self.tv_weight_stroke,
            "tv_loss_k": self.tv_loss_k,
            "curv_weight": self.curv_weight,
            "optimize_width": self.optimize_width
        }

class StrokeOptimRequest:
    def __init__(self, content_img, params, use_comet=False, style_img=None, experiment_name=None, experiment_tags=None): # Added style_img, experiment_name, experiment_tags
        self.content_img = content_img
        self.params = params
        self.use_comet = use_comet
        self.style_img = style_img # Added
        self.experiment_name = experiment_name # Added
        self.experiment_tags = experiment_tags # Added

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
        if self.style_img: # Conditionally add style_img
            payload["input"]["style_img"] = self.style_img
        if self.experiment_name: # Conditionally add experiment_name
            payload["input"]["experiment_name"] = self.experiment_name
        if self.experiment_tags: # Conditionally add experiment_tags
            payload["input"]["experiment_tags"] = self.experiment_tags

        # --- Debugging: Print full API request payload (conditional) ---
        print(f"API Request Payload: {json.dumps(payload, indent=2)}")
        # --- End Debugging ---

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
                "optimizer_lr_stroke": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "optimizer_color_lr": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "num_steps_stroke": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "num_strokes": ("INT", {"default": 3500, "min": 1, "max": 10000}),
                "width_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "length_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "print_freq": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "scale_by_y": ("BOOLEAN", {"default": False}),
                "init": (["random", "slic"], {"default": "slic"}),
                "verbose_output": ("BOOLEAN", {"default": False}),
                # Added parameters from handler.py analysis
                "img_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}), # Common image size
                "samples_per_curve": ("INT", {"default": 10, "min": 1, "max": 100}),
                "brushes_per_pixel": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "canvas_color": ("STRING", {"default": "black"}), # Or a list of colors, or RGB tuple
                "style_weight_stroke": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "content_weight_stroke": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tv_weight_stroke": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "tv_loss_k": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "curv_weight": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "optimize_width": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "style_image": ("IMAGE",), # Added style_image input
                "experiment_name": ("STRING", {"multiline": False, "default": ""}), # Added
                "experiment_tags": ("STRING", {"multiline": False, "default": ""}), # Added (comma-separated)
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
                                        scale_by_y, init, verbose_output,
                                        # Added parameters
                                        img_size, samples_per_curve, brushes_per_pixel, canvas_color,
                                        style_weight_stroke, content_weight_stroke, tv_weight_stroke,
                                        tv_loss_k, curv_weight, optimize_width,
                                        style_image=None, experiment_name=None, experiment_tags=None, use_comet=False):
        
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

        style_img_base64 = None
        if style_image is not None and style_image.numel() > 0:
            s_i = 255. * style_image.cpu().numpy().squeeze()
            style_img_pil = Image.fromarray(np.uint8(s_i))
            style_buffered = io.BytesIO()
            style_img_pil.save(style_buffered, format="PNG")
            style_img_base64 = base64.b64encode(style_buffered.getvalue()).decode('utf-8')

        # Convert comma-separated tags string to a list
        tags_list = [tag.strip() for tag in experiment_tags.split(',') if tag.strip()] if experiment_tags else None

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
            init=init,
            # Added parameters
            img_size=img_size,
            samples_per_curve=samples_per_curve,
            brushes_per_pixel=brushes_per_pixel,
            canvas_color=canvas_color,
            style_weight_stroke=style_weight_stroke,
            content_weight_stroke=content_weight_stroke,
            tv_weight_stroke=tv_weight_stroke,
            tv_loss_k=tv_loss_k,
            curv_weight=curv_weight,
            optimize_width=optimize_width
        )

        # 3. Create StrokeOptimRequest and submit job
        request_obj = StrokeOptimRequest(
            content_img=content_img_base64,
            params=params,
            use_comet=use_comet,
            style_img=style_img_base64, # Pass style_img
            experiment_name=experiment_name, # Pass experiment_name
            experiment_tags=tags_list # Pass experiment_tags as list
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
            
            # Try Gzip decompression first
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
    "SaveDecodedSVG": "Save Decoded SVG",
}