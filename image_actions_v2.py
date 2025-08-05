"""
Image Actions V2 for RL Logging Board

Unified image action system using PointToolBoxV6 for consistent processing
between training and visualization. Cleaner, simpler implementation.
"""

import os
import re
import json
import sys
from typing import List, Tuple, Optional, Dict, Any, Union
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


def display_image_with_actions_v2(image_path: str, response_text: str = "", **kwargs):
    """Display image using V2 unified processing with PointToolBoxV6.

    Args:
        image_path: Path to the image file
        response_text: The response text that might contain relevant information
        **kwargs: Additional arguments (for compatibility)
    """
    if not image_path:
        st.info('No image path provided.')
        return

    if not os.path.exists(image_path):
        st.info(f'Image not found: {image_path}')
        return

    # Create processor and process the image
    processor = ImageProcessorV2()
    result = processor.process_image_with_response(image_path, response_text)

    if not result["success"]:
        st.error(f"Error processing image: {result['error']}")
        return

    # Display UI controls
    with st.expander("🎨 Image Display Options", expanded=True):
        show_original = st.checkbox(
            "Show original image first",
            value=True,
            help="Display the original image before processed results"
        )

        chain_actions = st.checkbox(
            "Chain actions",
            value=False,
            help="Apply each action to the result of the previous action instead of the original image"
        )

        images_per_row = st.slider(
            "Images per row",
            min_value=1,
            max_value=4,
            value=2,
            help="Number of images to display per row"
        )

        show_errors = st.checkbox(
            "Show error details",
            value=False,
            help="Display detailed error information for failed tool calls"
        )

    # Show original image if requested
    if show_original:
        st.markdown("**Original Image**")
        st.image(result["original_image"], use_container_width=True)
        if result["processed_steps"]:
            st.divider()

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
def display_image_with_actions(image_path: str, response_text: str = "", **kwargs):
    """Backward compatibility alias for the V2 function."""
    return display_image_with_actions_v2(image_path, response_text, **kwargs)


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