"""
Image Actions V2 for RL Logging Board

Unified image action system using PointToolBoxV6 for consistent processing
between training and visualization. Cleaner, simpler implementation.
"""

import os
import re
import json
import sys
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence
from PIL import Image

import streamlit as st

# Import PointToolBoxV6 from the verl package
import importlib.util
verl_spec = importlib.util.find_spec("verl")
if verl_spec is None:
    raise ImportError("verl package not found. Please ensure it's installed.")

verl_path = verl_spec.submodule_search_locations[0]
sys.path.append(os.path.join(verl_path, "workers", "agent", "envs", "mm_process_engine"))
from point_toolbox_v6 import PointToolBoxV6

from path_utils import resolve_image_path


def resize_image_max_edge(image: Image.Image, max_edge: int = 200) -> Image.Image:
    """Resize image so that the maximum edge is max_edge pixels, maintaining aspect ratio.

    Args:
        image: PIL Image object
        max_edge: Maximum edge length in pixels

    Returns:
        Resized PIL Image object
    """
    width, height = image.size

    # If both edges are already smaller than max_edge, return original
    if width <= max_edge and height <= max_edge:
        return image

    # Calculate scaling factor
    if width > height:
        scale = max_edge / width
    else:
        scale = max_edge / height

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image using high-quality downsampling
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class ImageProcessorV2:
    """Unified image processor using PointToolBoxV6 for consistent tool execution."""

    def __init__(self):
        self.toolbox = PointToolBoxV6(
            _name="visualization_toolbox",
            _desc="Tool for image visualization processing",
            _params={},
            on_processed_image=True,
            record_processed_image=True
        )
        self.processed_steps = []

    def extract_tool_calls_in_order(self, response_text: str) -> List[Tuple[str, dict, int]]:
        """Extract all tool calls from response text in the order they appear.

        Args:
            response_text: The response text containing tool calls

        Returns:
            List of tuples: [(tool_name, tool_args, position), ...]
        """
        if not response_text:
            return []

        tool_calls = []

        # Extract tool call blocks using regex with position tracking
        pattern = r'<tool_call>(.*?)</tool_call>'
        for match in re.finditer(pattern, response_text, re.DOTALL):
            try:
                tool_call_str = match.group(1).strip()
                tool_call = json.loads(tool_call_str)

                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("arguments", {})
                position = match.start()  # Position in text

                if tool_name:
                    tool_calls.append((tool_name, tool_args, position))

            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Skip invalid tool calls
                continue

        # Sort by position to maintain order
        tool_calls.sort(key=lambda x: x[2])
        return tool_calls

    def process_image_with_response(self, image_path: str, response_text: str) -> Dict[str, Any]:
        """Process image with tool calls from response text.

        Args:
            image_path: Path to the image file
            response_text: Response text containing tool calls

        Returns:
            Dict containing processing results and metadata
        """
        # Load the image
        try:
            image = Image.open(image_path)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load image: {e}",
                "original_image": None,
                "processed_steps": [],
                "tool_calls": []
            }

        # Reset toolbox with the image
        try:
            self.toolbox.reset(
                raw_prompt="Visualization processing",
                multi_modal_data={'image': [image]},
                origin_multi_modal_data={'image': [image]},
                on_processed_image=True,
                record_processed_image=True
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize toolbox: {e}",
                "original_image": image,
                "processed_steps": [],
                "tool_calls": []
            }

        # Extract tool calls
        tool_calls = self.extract_tool_calls_in_order(response_text)

        if not tool_calls:
            return {
                "success": True,
                "error": None,
                "original_image": image,
                "processed_steps": [],
                "tool_calls": []
            }

        processed_steps = []

        # Process each tool call
        for i, (tool_name, tool_args, position) in enumerate(tool_calls):
            try:
                # Create action string for toolbox
                action_string = f'<tool_call>{json.dumps({"name": tool_name, "arguments": tool_args})}</tool_call>'

                # Execute using toolbox
                result_image, obs, reward, done, info = self.toolbox.execute(action_string, return_image=True)

                if result_image is not None and info.get("status") == "success":
                    # Success
                    tool_display_name = tool_name.replace('_', ' ').replace('image ', '').title()
                    processed_steps.append({
                        "step": i + 1,
                        "tool_name": tool_name,
                        "tool_display_name": tool_display_name,
                        "tool_args": tool_args,
                        "result_image": result_image,
                        "success": True,
                        "error": None,
                        "caption": self._create_caption(tool_name, tool_args, i + 1)
                    })
                else:
                    # Error occurred
                    error_msg = info.get("error", "Unknown error")
                    user_prompt = obs if isinstance(obs, str) else str(obs)
                    processed_steps.append({
                        "step": i + 1,
                        "tool_name": tool_name,
                        "tool_display_name": tool_name.replace('_', ' ').replace('image ', '').title(),
                        "tool_args": tool_args,
                        "result_image": None,
                        "success": False,
                        "error": error_msg,
                        "user_prompt": user_prompt,
                        "caption": f"Error in {tool_name}"
                    })

            except Exception as e:
                # Unexpected error
                processed_steps.append({
                    "step": i + 1,
                    "tool_name": tool_name,
                    "tool_display_name": tool_name.replace('_', ' ').replace('image ', '').title(),
                    "tool_args": tool_args,
                    "result_image": None,
                    "success": False,
                    "error": str(e),
                    "user_prompt": f"Unexpected error: {e}",
                    "caption": f"Error in {tool_name}"
                })

        return {
            "success": True,
            "error": None,
            "original_image": image,
            "processed_steps": processed_steps,
            "tool_calls": tool_calls
        }

    def _create_caption(self, tool_name: str, tool_args: dict, step: int) -> str:
        """Create a descriptive caption for the processed image."""
        if tool_name == "image_zoom_in_tool":
            bbox = tool_args.get("bbox_2d", [])
            if len(bbox) >= 4:
                return f"Zoom {step}: [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
            return f"Zoom {step}"
        elif tool_name == "image_mark_points_tool":
            points = tool_args.get("point_2d", [])
            label = tool_args.get("label", "")
            if isinstance(points, list) and len(points) > 0:
                if isinstance(points[0], list):
                    # Multiple points
                    count = len(points)
                    caption = f"Marked {count} points"
                    if label:
                        caption += f": {label}"
                    return caption
                else:
                    # Single point
                    if label:
                        return f"Point at {points}: {label}"
                    else:
                        return f"Point at {points}"
            return f"Mark Points {step}"
        elif tool_name == "image_draw_line_tool":
            points = tool_args.get("points", [])
            return f"Line {step}: {len(points)} points"
        elif tool_name == "image_draw_horizontal_line_tool":
            y_pos = tool_args.get("y_position", "?")
            style = tool_args.get("style", "solid")
            return f"Horizontal Line {step}: y={y_pos} ({style})"
        elif tool_name == "image_draw_vertical_line_tool":
            x_pos = tool_args.get("x_position", "?")
            style = tool_args.get("style", "solid")
            return f"Vertical Line {step}: x={x_pos} ({style})"
        else:
            return f"{tool_name.replace('_', ' ').title()} {step}"


def display_processed_images_from_jsonl(
    processed_images: List[Dict[str, str]],
    images_per_row: int,
    max_edge: int = 200,
    show_original: bool = False,
    log_dir: Optional[str] = None,
    search_subdirs: Optional[Sequence[str]] = None,
    extra_search_paths: Optional[Sequence[str]] = None,
):
    """Display processed images from JSONL data.

    Args:
        processed_images: List of processed image entries from JSONL
        images_per_row: Number of images to display per row
        max_edge: Maximum edge length for resizing images (default: 200)
        show_original: Whether to display origin images (default: False)
    """
    if not processed_images:
        st.info("No processed images to display.")
        return

    # Separate origin and processed images
    origin_images = [img for img in processed_images if img.get('tool') == 'origin']
    processed_imgs = [img for img in processed_images if img.get('tool') != 'origin']

    st.markdown("**Processed Images from JSONL**")

    # Show statistics
    st.caption(f"Total: {len(processed_images)} images | Origin: {len(origin_images)} | Processed: {len(processed_imgs)}")

    # Prepare images for display
    display_images = []

    # Add origin images first only if show_original is True
    if show_original:
        for img in origin_images:
            resolved_path, attempted = resolve_image_path(
                img.get('path', ''),
                log_dir=log_dir,
                search_subdirs=search_subdirs,
                extra_search_paths=extra_search_paths,
            )
            if resolved_path and os.path.exists(resolved_path):
                display_images.append({
                    'path': resolved_path,
                    'caption': f"Original ({img['tool']})",
                    'tool': img['tool']
                })
            else:
                suffix = f" (checked: {attempted[-1]})" if attempted else ""
                st.warning(f"Origin image not found: {img.get('path', '')}{suffix}")

    # Add processed images
    for i, img in enumerate(processed_imgs):
        resolved_path, attempted = resolve_image_path(
            img.get('path', ''),
            log_dir=log_dir,
            search_subdirs=search_subdirs,
            extra_search_paths=extra_search_paths,
        )
        if resolved_path and os.path.exists(resolved_path):
            tool_name = img['tool'].replace('_', ' ').title()

            # Check for label in tool arguments for zoom-in tools
            if img['tool'] == 'image_zoom_in_tool':
                label = img.get('label', '')  # Get label from stored tool arguments
                if label:
                    caption = f"Step {i+1}: {label}"
                else:
                    # Fallback to bbox coordinates if no label
                    bbox = img.get('bbox_2d', [])
                    if len(bbox) >= 4:
                        caption = f"Step {i+1}: [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
                    else:
                        caption = f"Step {i+1}: {tool_name}"
            else:
                # For non-zoom tools, use the original format
                caption = f"Step {i+1}: {tool_name}"

            display_images.append({
                'path': resolved_path,
                'caption': caption,
                'tool': img['tool']
            })
        else:
            suffix = f" (checked: {attempted[-1]})" if attempted else ""
            st.warning(f"Processed image not found: {img.get('path', '')}{suffix}")

    if not display_images:
        st.error("No valid image files found in processed_images paths.")
        return

    # Display images in rows
    for i in range(0, len(display_images), images_per_row):
        cols = st.columns(images_per_row)

        for j in range(images_per_row):
            idx = i + j
            if idx < len(display_images):
                img_info = display_images[idx]

                with cols[j]:
                    try:
                        image = Image.open(img_info['path'])
                        # Resize image for faster web loading
                        image = resize_image_max_edge(image, max_edge=max_edge)
                        st.image(image, caption=img_info['caption'], use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading {img_info['caption']}: {e}")


def display_image_with_actions_v2(
    image_path: str,
    response_text: str = "",
    processed_images: Optional[List[Dict[str, str]]] = None,
    *,
    log_dir: Optional[str] = None,
    search_subdirs: Optional[Sequence[str]] = None,
    extra_search_paths: Optional[Sequence[str]] = None,
    **kwargs
):
    """Display image using V2 unified processing with PointToolBoxV6.

    Args:
        image_path: Path to the image file
        response_text: The response text that might contain relevant information
        processed_images: List of processed image entries from JSONL (e.g., [{"tool": "origin", "path": "..."}, {"tool": "zoom_in", "path": "..."}])
        **kwargs: Additional arguments (for compatibility)
    """
    if not image_path:
        st.info('No image path provided.')
        return

    effective_image_path = image_path
    attempted_paths: List[str] = []
    if not os.path.exists(effective_image_path):
        resolved_path, attempted_paths = resolve_image_path(
            image_path,
            log_dir=log_dir,
            search_subdirs=search_subdirs,
            extra_search_paths=extra_search_paths,
        )
        if resolved_path:
            effective_image_path = resolved_path

    if not os.path.exists(effective_image_path):
        suffix = f' (checked: {attempted_paths[-1]})' if attempted_paths else ''
        st.info(f'Image not found: {image_path}{suffix}')
        return

    # Check if we have processed_images data
    has_processed_images = (
        processed_images is not None
        and isinstance(processed_images, list)
        and len(processed_images) > 0
        and all(isinstance(item, dict) and 'tool' in item and 'path' in item for item in processed_images)
    )

    # Load original image (lightweight operation)
    try:
        original_image = Image.open(effective_image_path)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return

    # Display UI controls - determine mode first
    with st.expander("🎨 Image Display Options", expanded=True):
        # Choose display mode
        if has_processed_images:
            use_processed_images = st.checkbox(
                "Use processed images from JSONL",
                value=True,
                help="Use pre-processed images from the JSONL file instead of processing response text"
            )
            if use_processed_images:
                st.info(f"Found {len(processed_images)} processed images in JSONL data")
        else:
            use_processed_images = False
            if processed_images is not None:
                st.warning("Processed images data found but invalid format. Using response text processing.")

        # Default to False if using processed images (since original is likely included in processed images)
        show_original_default = not use_processed_images
        show_original = st.checkbox(
            "Show original image first",
            value=show_original_default,
            help="Display the original image before processed results"
        )

        images_per_row = st.slider(
            "Images per row",
            min_value=1,
            max_value=4,
            value=2,
            help="Number of images to display per row"
        )

        max_edge = st.slider(
            "Maximum image size (pixels)",
            min_value=100,
            max_value=1000,
            value=200,
            step=50,
            help="Maximum edge length for resizing images (larger = higher quality but slower loading)"
        )

        # Only show these controls if NOT using processed images
        if not use_processed_images:
            chain_actions = st.checkbox(
                "Chain actions",
                value=False,
                help="Apply each action to the result of the previous action instead of the original image"
            )
            show_errors = st.checkbox(
                "Show error details",
                value=False,
                help="Display detailed error information for failed tool calls"
            )
        else:
            chain_actions = False  # Not applicable for processed images
            show_errors = False  # Not applicable for processed images

    # Show original image if requested
    if show_original:
        st.markdown("**Original Image**")
        st.image(original_image, use_container_width=True)
        st.divider()

    # OPTIMIZATION: Early return for processed images path - skip toolbox entirely!
    if use_processed_images and has_processed_images:
        display_processed_images_from_jsonl(
            processed_images,
            images_per_row,
            max_edge,
            show_original,
            log_dir=log_dir,
            search_subdirs=search_subdirs,
            extra_search_paths=extra_search_paths,
        )
        return

    # Only create processor and process when actually needed (NOT using processed_images)
    processor = ImageProcessorV2()
    result = processor.process_image_with_response(effective_image_path, response_text)

    if not result["success"]:
        st.error(f"Error processing image: {result['error']}")
        return

    # Process and display results
    processed_steps = result["processed_steps"]

    if not processed_steps:
        st.info("No tool calls found in the response text.")
        return

    # Prepare results for display
    results_to_display = []
    current_image = result["original_image"]

    for step in processed_steps:
        if step["success"]:
            # Successful tool execution
            if chain_actions:
                # Use the result from this step for next step
                display_image = step["result_image"]
                current_image = step["result_image"]
            else:
                # Always use result from original image
                display_image = step["result_image"]

            results_to_display.append((
                display_image,
                step["caption"],
                step["tool_name"],
                step["success"],
                step.get("error"),
                step.get("user_prompt")
            ))
        else:
            # Failed tool execution - show error
            if show_errors:
                results_to_display.append((
                    None,  # No image
                    step["caption"],
                    step["tool_name"],
                    step["success"],
                    step.get("error"),
                    step.get("user_prompt")
                ))

    # Display results
    if results_to_display:
        action_mode = "Chained" if chain_actions else "Sequential"
        st.markdown(f"**Tool Results ({action_mode} - in order of occurrence)**")

        # Display images in rows
        for i in range(0, len(results_to_display), images_per_row):
            cols = st.columns(images_per_row)

            for j in range(images_per_row):
                idx = i + j
                if idx < len(results_to_display):
                    image, caption, tool_name, success, error, user_prompt = results_to_display[idx]

                    with cols[j]:
                        if success and image is not None:
                            # Successful result with image
                            st.image(image, caption=caption, use_container_width=True)
                        elif not success:
                            # Error result
                            st.error(f"❌ {caption}")
                            if show_errors and error:
                                st.write(f"**Error:** {error}")
                            if show_errors and user_prompt:
                                st.write(f"**User Prompt:** {user_prompt}")
                        else:
                            # Unexpected case
                            st.warning(f"⚠️ {caption}: No result image")
    else:
        st.info("No valid tool calls found that can be processed.")


# Alias for backward compatibility
def display_image_with_actions(image_path: str, response_text: str = "", processed_images: Optional[List[Dict[str, str]]] = None, **kwargs):
    """Backward compatibility alias for the V2 function."""
    return display_image_with_actions_v2(image_path, response_text, processed_images=processed_images, **kwargs)


if __name__ == "__main__":
    # Test the V2 implementation
    st.title("Image Actions V2 Test")

    # Test data
    test_image_path = "/scratch/doqihu/laughing-potato/real_case/2/image.png"
    test_response = """
Let me analyze this form by zooming into specific sections.

<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [84, 485, 755, 546], "label": "Applicant's Name"}}
</tool_call>

Now let me mark some key points:

<tool_call>
{"name": "image_mark_points_tool", "arguments": {"point_2d": [[100, 100], [200, 200]], "label": "Key Points", "color": [255, 0, 0]}}
</tool_call>
"""

    if os.path.exists(test_image_path):
        st.write("Testing with real image:")
        display_image_with_actions_v2(test_image_path, test_response)
    else:
        st.info("Test image not found. Upload an image to test the V2 functionality.")

        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write("Testing with uploaded image:")
            display_image_with_actions_v2(temp_path, test_response)

            # Clean up
            os.remove(temp_path)
