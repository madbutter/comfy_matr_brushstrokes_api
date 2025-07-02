# Matr Brushstrokes ComfyUI Node

This custom ComfyUI node allows you to integrate with the `matr-brushstrokes-api` endpoint to convert images into vector-based brushstroke art (SVG) and a rasterized image representation.

## Installation

1.  **Locate your ComfyUI `custom_nodes` directory**: This is typically found within your ComfyUI installation folder (e.g., `ComfyUI/custom_nodes/`).
2.  **Place the node**: Copy the entire `comfy_matr_brushstrokes_node` directory into the `custom_nodes` directory.
3.  **Restart ComfyUI**: Close and restart your ComfyUI application to allow it to discover the new node.

## Usage

1.  **Add the Node**: In the ComfyUI interface, right-click on the canvas, navigate to the `Image/MatrBrushstrokes` category, and select "Matr Brushstrokes API".
2.  **Connect Inputs**:
    *   **`image`**: Connect an `IMAGE` input (e.g., from a "Load Image" node).
    *   **`api_key`**: Enter your `matr-brushstrokes-api` Runpod API Key.
    *   **`base_url`**: The default URL is `https://api.runpod.ai/v2/vt0tvhzcqug7zu`. Only change if your endpoint differs.
    *   **Parameters**: Adjust the various brushstroke optimization parameters (`optimizer_lr_stroke`, `num_steps_stroke`, `num_strokes`, `width_scale`, `length_scale`, `print_freq`, `scale_by_y`, `init`) as needed.
    *   **`use_comet`**: Optional boolean to enable Comet integration (default: `False`).
3.  **Connect Outputs**:
    *   **`SVG_Output`**: Connect this to a "Save Text" node or similar to save the generated SVG string.
    *   **`Image_Output`**: Connect this to a "Save Image" node to save the rasterized image.
4.  **Queue Prompt**: Execute the workflow to generate the brushstroke art.

## Node Details

### Inputs

*   **`image`** (`IMAGE`): The input image to be processed.
*   **`api_key`** (`STRING`): Your Runpod API Key for `matr-brushstrokes-api`.
*   **`base_url`** (`STRING`): The base URL of the `matr-brushstrokes-api` endpoint. Default: `https://api.runpod.ai/v2/vt0tvhzcqug7zu`.
*   **`optimizer_lr_stroke`** (`FLOAT`): Learning rate for stroke optimization. Default: `0.1`.
*   **`optimizer_color_lr`** (`FLOAT`): Learning rate for color optimization. Default: `0.001`.
*   **`num_steps_stroke`** (`INT`): Number of optimization steps for strokes. Default: `5001`.
*   **`num_strokes`** (`INT`): Number of strokes to generate. Default: `250`.
*   **`width_scale`** (`FLOAT`): Scaling factor for stroke width. Default: `5.0`.
*   **`length_scale`** (`FLOAT`): Scaling factor for stroke length. Default: `0.1`.
*   **`print_freq`** (`INT`): Frequency for printing intermediate results. Default: `50`.
*   **`scale_by_y`** (`BOOLEAN`): Boolean to scale by Y-axis. Default: `True`.
*   **`init`** (`STRING`): Initialization method for strokes. Options: `random`, `content`. Default: `random`.
*   **`use_comet`** (`BOOLEAN`, Optional): Enable Comet integration. Default: `False`.

### Outputs

*   **`SVG_Output`** (`STRING`): The generated SVG string representing the vector analysis of the image.
*   **`Image_Output`** (`IMAGE`): The rasterized PNG image derived from the vector analysis.

## Troubleshooting

*   **API Key Missing/Invalid**: Ensure your `api_key` is correctly entered. Check the ComfyUI console for `ValueError: API Key is required.` or `API Request failed` messages.
*   **Network Issues**: If the API endpoint is unreachable, you might see `API Request failed` messages. Verify your internet connection and the `base_url`.
*   **Job Failure**: If the `matr-brushstrokes-api` job fails, the node will raise a `RuntimeError`. Check the ComfyUI console for details.
*   **Image Format**: Ensure the input image is compatible. The node converts ComfyUI `IMAGE` (PyTorch tensor) to PNG for API submission.