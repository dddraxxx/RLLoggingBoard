"""
Image Actions for RL Logging Board

Simplified modular image action system with unified rendering.
Uses functions from actions.py for image processing.
"""

import os
import re
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Union

import streamlit as st
from PIL import Image

# Import functions from the actions toolbox
import sys
import importlib.util
verl_spec = importlib.util.find_spec("verl")
verl_path = verl_spec.submodule_search_locations[0]
actions_path = os.path.join(verl_path, "toolbox")
sys.path.append(actions_path)
from actions import (
    execute_zoom_in_tool,
    execute_mark_points_tool,
    execute_draw_line_tool,
    execute_draw_horizontal_line_tool,
    execute_draw_vertical_line_tool,
)


class ImageAction(ABC):
    """Base class for image visualization actions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the action."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this action does."""
        pass

    @abstractmethod
    def execute(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> List[Tuple[Image.Image, str]]:
        """Execute the image action.

        Args:
            image_input: Either a path to the image file or a PIL Image object
            response_text: The response text that might contain relevant information
            **kwargs: Additional arguments that might be needed

        Returns:
            List of (image, caption) tuples representing the processed results
        """
        pass

    @abstractmethod
    def is_applicable(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> bool:
        """Check if this action is applicable for the given inputs.

        Args:
            image_input: Either a path to the image file or a PIL Image object
            response_text: The response text that might contain relevant information
            **kwargs: Additional arguments that might be needed

        Returns:
            True if this action can be executed with the given inputs
        """
        pass

    def _render_image_grid(self, images_with_captions: List[Tuple[Image.Image, str]], cols: int = 2) -> None:
        """Unified rendering method for displaying images in a grid layout.

        Args:
            images_with_captions: List of (image, caption) tuples
            cols: Number of columns in the grid
        """
        if not images_with_captions:
            return

        for idx in range(0, len(images_with_captions), cols):
            columns = st.columns(cols)

            for col_idx in range(cols):
                img_idx = idx + col_idx
                if img_idx < len(images_with_captions):
                    image, caption = images_with_captions[img_idx]
                    with columns[col_idx]:
                        st.image(image, caption=caption, use_container_width=True)

    def _load_image(self, image_input: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Load an image from either a file path or PIL Image object.

        Args:
            image_input: Either a path to the image file or a PIL Image object

        Returns:
            PIL Image object or None if loading fails
        """
        try:
            if isinstance(image_input, str):
                return Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                return image_input
            else:
                st.error(f"Invalid image input type: {type(image_input)}")
                return None
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None


class ShowOriginalImageAction(ImageAction):
    """Action to display the original image."""

    @property
    def name(self) -> str:
        return "Show Original Image"

    @property
    def description(self) -> str:
        return "Display the original image without any modifications"

    def execute(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> List[Tuple[Image.Image, str]]:
        """Display the original image."""
        image = self._load_image(image_input)
        if image:
            if isinstance(image_input, str):
                caption = f"Image: {os.path.basename(image_input)}"
            else:
                caption = "Original Image"
            return [(image, caption)]
        return []

    def is_applicable(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> bool:
        """Always applicable if image input is valid."""
        if isinstance(image_input, str):
            return bool(image_input and os.path.exists(image_input))
        elif isinstance(image_input, Image.Image):
            return True
        return False


class DetectAndCropBoxesAction(ImageAction):
    """Action to detect and crop boxes from image_zoom_in_tool calls using actions.py."""
    tool_name = ["image_zoom_in_tool"]

    @property
    def name(self) -> str:
        return "Detect and Crop Boxes"

    @property
    def description(self) -> str:
        return "Extract box coordinates from image_zoom_in_tool calls and display cropped regions"

    def execute(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> List[Tuple[Image.Image, str]]:
        """Extract tool calls and return cropped regions."""
        image = self._load_image(image_input)
        if not image:
            return []

        tool_calls = self._extract_zoom_in_tool_calls(response_text)

        if not tool_calls:
            return []

        cropped_images = []
        for idx, coords in enumerate(tool_calls):
            try:
                # Use the execute_zoom_in_tool from actions.py
                cropped_img = execute_zoom_in_tool(
                    image,
                    {"bbox_2d": coords},
                    image.size[0],
                    image.size[1]
                )
                caption = f"Box {idx+1}: [{coords[0]},{coords[1]},{coords[2]},{coords[3]}]"
                cropped_images.append((cropped_img, caption))
            except Exception as e:
                st.warning(f"Error cropping box {idx+1}: {e}")

        return cropped_images

    def is_applicable(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> bool:
        """Applicable if image exists and response contains image_zoom_in_tool calls."""
        if isinstance(image_input, str):
            if not (image_input and os.path.exists(image_input) and response_text):
                return False
        elif isinstance(image_input, Image.Image):
            if not response_text:
                return False
        else:
            return False

        tool_calls = self._extract_zoom_in_tool_calls(response_text)
        return len(tool_calls) > 0

    def _extract_zoom_in_tool_calls(self, text: str) -> List[List[int]]:
        """Extract image_zoom_in_tool calls from text.

        Returns:
            List of bbox coordinates: [[x1, y1, x2, y2], ...]
        """
        if not text:
            return []

        tool_calls = []

        # Extract tool call blocks using regex
        tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)

        for tool_call_str in tool_call_matches:
            try:
                # Parse JSON
                tool_call = json.loads(tool_call_str.strip())

                # Check if it's an image_zoom_in_tool call
                if tool_call.get("name") in self.tool_name:
                    args = tool_call.get("arguments", {})
                    bbox = args.get("bbox_2d", [])

                    # Validate bbox format
                    if (isinstance(bbox, (list, tuple)) and
                        len(bbox) >= 4 and
                        all(isinstance(coord, (int, float)) for coord in bbox[:4])):

                        # Convert to integers for consistency
                        bbox_coords = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                        tool_calls.append(bbox_coords)

            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Skip invalid tool calls
                continue

        return tool_calls


class MarkPointAction(ImageAction):
    """Action to detect and mark points from image_mark_points_tool calls using actions.py."""
    tool_name = ["image_mark_points_tool"]
    @property
    def name(self) -> str:
        return "Mark Points"

    @property
    def description(self) -> str:
        return "Detect image_mark_points_tool calls and visualize marked points on the image"

    def execute(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> List[Tuple[Image.Image, str]]:
        """Extract tool calls and return images with marked points."""
        image = self._load_image(image_input)
        if not image:
            return []

        tool_calls = self._extract_mark_points_tool_calls(response_text)

        if not tool_calls:
            return []

        marked_images = []
        for idx, (points, label, color, size, shape) in enumerate(tool_calls):
            try:
                # Use the execute_mark_points_tool from actions.py
                args = {
                    "point_2d": points,
                    "color": color,
                    "size": size,
                    "shape": shape,
                    "label": label
                }
                marked_image = execute_mark_points_tool(image, args)

                if label:
                    caption = f"Point at {points}: {label}"
                else:
                    caption = f"Point at {points}"

                marked_images.append((marked_image, caption))
            except Exception as e:
                st.error(f"Error processing point {idx+1}: {e}")

        return marked_images

    def is_applicable(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> bool:
        """Applicable if image exists and response contains image_mark_points_tool calls."""
        if isinstance(image_input, str):
            if not (image_input and os.path.exists(image_input) and response_text):
                return False
        elif isinstance(image_input, Image.Image):
            if not response_text:
                return False
        else:
            return False

        tool_calls = self._extract_mark_points_tool_calls(response_text)
        return len(tool_calls) > 0

    def _extract_mark_points_tool_calls(self, text: str) -> List[Tuple[List[int], str, Tuple[int, int, int], int, str]]:
        """Extract image_mark_points_tool calls from text.

        Returns:
            List of tuples: [(points, label, color, size, shape), ...]
        """
        if not text:
            return []

        tool_calls = []

        # Extract tool call blocks using regex
        tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)

        for tool_call_str in tool_call_matches:
            try:
                # Parse JSON
                tool_call = json.loads(tool_call_str.strip())

                # Check if it's an image_mark_points_tool call
                if tool_call.get("name") in self.tool_name:
                    args = tool_call.get("arguments", {})
                    points = args.get("point_2d", [])
                    label = args.get("label", "")
                    color = args.get("color", (255, 0, 0))  # Default red
                    size = args.get("size", 5)  # Default size
                    shape = args.get("shape", "X")  # Default shape

                    # Validate points format
                    if (isinstance(points, (list, tuple)) and
                        len(points) >= 2 and
                        all(isinstance(coord, (int, float)) for coord in points[:2])):

                        # Convert to integers for consistency
                        point_coords = [int(points[0]), int(points[1])]

                        # Ensure color is tuple
                        if isinstance(color, list):
                            color = tuple(color)
                        elif isinstance(color, str):
                            color = (255, 0, 0)  # Default if string

                        tool_calls.append((point_coords, label, color, size, shape))

            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Skip invalid tool calls
                continue

        return tool_calls


class DrawLineAction(ImageAction):
    """Action to detect and draw lines from draw_line_tool calls using actions.py."""
    tool_name = ["image_draw_line_tool"]

    @property
    def name(self) -> str:
        return "Draw Lines"

    @property
    def description(self) -> str:
        return "Detect draw_line_tool calls and draw lines on the image"

    def execute(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> List[Tuple[Image.Image, str]]:
        """Extract tool calls and return images with drawn lines."""
        image = self._load_image(image_input)
        if not image:
            return []

        tool_calls = self._extract_draw_line_tool_calls(response_text)

        if not tool_calls:
            return []

        line_images = []
        for idx, (points, color, thickness) in enumerate(tool_calls):
            try:
                # Use the execute_draw_line_tool from actions.py
                args = {
                    "points": points,
                    "color": color,
                    "thickness": thickness
                }
                line_image = execute_draw_line_tool(image, args)

                caption = f"Line {idx+1}: {len(points)} points"
                line_images.append((line_image, caption))
            except Exception as e:
                st.error(f"Error processing line {idx+1}: {e}")

        return line_images

    def is_applicable(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> bool:
        """Applicable if image exists and response contains draw_line_tool calls."""
        if isinstance(image_input, str):
            if not (image_input and os.path.exists(image_input) and response_text):
                return False
        elif isinstance(image_input, Image.Image):
            if not response_text:
                return False
        else:
            return False

        tool_calls = self._extract_draw_line_tool_calls(response_text)
        return len(tool_calls) > 0

    def _extract_draw_line_tool_calls(self, text: str) -> List[Tuple[List[List[int]], Tuple[int, int, int], int]]:
        """Extract draw_line_tool calls from text.

        Returns:
            List of tuples: [(points, color, thickness), ...]
        """
        if not text:
            return []

        tool_calls = []

        # Extract tool call blocks using regex
        tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)

        for tool_call_str in tool_call_matches:
            try:
                # Parse JSON
                tool_call = json.loads(tool_call_str.strip())

                # Check if it's a draw_line_tool call
                if tool_call.get("name") in self.tool_name:
                    args = tool_call.get("arguments", {})
                    points = args.get("points", [])
                    color = args.get("color", (255, 0, 0))  # Default red
                    thickness = args.get("thickness", 3)  # Default thickness

                    # Validate points format
                    if (isinstance(points, list) and len(points) >= 2 and
                        all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in points)):

                        # Convert to proper format
                        formatted_points = [[int(p[0]), int(p[1])] for p in points]

                        # Ensure color is tuple
                        if isinstance(color, list):
                            color = tuple(color)
                        elif isinstance(color, str):
                            color = (255, 0, 0)  # Default if string

                        tool_calls.append((formatted_points, color, thickness))

            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Skip invalid tool calls
                continue

        return tool_calls


# ------------------ New Actions: Horizontal & Vertical Lines ------------------ #


class DrawHorizontalLineAction(ImageAction):
    """Action to detect and draw horizontal lines from image_draw_horizontal_line_tool calls."""

    tool_name = ["image_draw_horizontal_line_tool"]

    @property
    def name(self) -> str:
        return "Draw Horizontal Lines"

    @property
    def description(self) -> str:
        return "Detect image_draw_horizontal_line_tool calls and draw horizontal lines on the image"

    def execute(
        self,
        image_input: Union[str, Image.Image],
        response_text: str,
        **kwargs,
    ) -> List[Tuple[Image.Image, str]]:
        """Extract tool calls and return images with horizontal lines."""

        image = self._load_image(image_input)
        if not image:
            return []

        tool_calls = self._extract_draw_horizontal_line_tool_calls(response_text)

        if not tool_calls:
            return []

        line_images = []
        for idx, (y_pos, color, thickness, style) in enumerate(tool_calls):
            try:
                args = {
                    "y_position": y_pos,
                    "color": color,
                    "thickness": thickness,
                    "style": style,
                }
                line_image = execute_draw_horizontal_line_tool(image, args)
                caption = f"Horizontal Line {idx + 1}: y={y_pos} ({style})"
                line_images.append((line_image, caption))
            except Exception as e:
                st.error(f"Error processing horizontal line {idx + 1}: {e}")

        return line_images

    def is_applicable(
        self,
        image_input: Union[str, Image.Image],
        response_text: str,
        **kwargs,
    ) -> bool:
        if isinstance(image_input, str):
            if not (image_input and os.path.exists(image_input) and response_text):
                return False
        elif isinstance(image_input, Image.Image):
            if not response_text:
                return False
        else:
            return False

        tool_calls = self._extract_draw_horizontal_line_tool_calls(response_text)
        return len(tool_calls) > 0

    def _extract_draw_horizontal_line_tool_calls(
        self, text: str
    ) -> List[Tuple[int, Tuple[int, int, int], int, str]]:
        """Extract horizontal line tool calls from text.

        Returns:
            List of tuples: [(y_position, color, thickness, style), ...]
        """

        if not text:
            return []

        tool_calls = []

        tool_call_matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)

        for tool_call_str in tool_call_matches:
            try:
                tool_call = json.loads(tool_call_str.strip())

                if tool_call.get("name") in self.tool_name:
                    args = tool_call.get("arguments", {})
                    y_pos = args.get("y_position")
                    color = args.get("color", (255, 0, 0))
                    thickness = args.get("thickness", 3)
                    style = args.get("style", "solid")

                    if y_pos is None or not isinstance(y_pos, (int, float)):
                        continue

                    y_pos = int(y_pos)

                    if isinstance(color, list):
                        color = tuple(color)
                    elif isinstance(color, str):
                        color = (255, 0, 0)

                    tool_calls.append((y_pos, color, thickness, style))

            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        return tool_calls


class DrawVerticalLineAction(ImageAction):
    """Action to detect and draw vertical lines from image_draw_vertical_line_tool calls."""

    tool_name = ["image_draw_vertical_line_tool"]

    @property
    def name(self) -> str:
        return "Draw Vertical Lines"

    @property
    def description(self) -> str:
        return "Detect image_draw_vertical_line_tool calls and draw vertical lines on the image"

    def execute(
        self,
        image_input: Union[str, Image.Image],
        response_text: str,
        **kwargs,
    ) -> List[Tuple[Image.Image, str]]:
        """Extract tool calls and return images with vertical lines."""

        image = self._load_image(image_input)
        if not image:
            return []

        tool_calls = self._extract_draw_vertical_line_tool_calls(response_text)

        if not tool_calls:
            return []

        line_images = []
        for idx, (x_pos, color, thickness, style) in enumerate(tool_calls):
            try:
                args = {
                    "x_position": x_pos,
                    "color": color,
                    "thickness": thickness,
                    "style": style,
                }
                line_image = execute_draw_vertical_line_tool(image, args)
                caption = f"Vertical Line {idx + 1}: x={x_pos} ({style})"
                line_images.append((line_image, caption))
            except Exception as e:
                st.error(f"Error processing vertical line {idx + 1}: {e}")

        return line_images

    def is_applicable(
        self,
        image_input: Union[str, Image.Image],
        response_text: str,
        **kwargs,
    ) -> bool:
        if isinstance(image_input, str):
            if not (image_input and os.path.exists(image_input) and response_text):
                return False
        elif isinstance(image_input, Image.Image):
            if not response_text:
                return False
        else:
            return False

        tool_calls = self._extract_draw_vertical_line_tool_calls(response_text)
        return len(tool_calls) > 0

    def _extract_draw_vertical_line_tool_calls(
        self, text: str
    ) -> List[Tuple[int, Tuple[int, int, int], int, str]]:
        """Extract vertical line tool calls from text.

        Returns:
            List of tuples: [(x_position, color, thickness, style), ...]
        """

        if not text:
            return []

        tool_calls = []

        tool_call_matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)

        for tool_call_str in tool_call_matches:
            try:
                tool_call = json.loads(tool_call_str.strip())

                if tool_call.get("name") in self.tool_name:
                    args = tool_call.get("arguments", {})
                    x_pos = args.get("x_position")
                    color = args.get("color", (255, 0, 0))
                    thickness = args.get("thickness", 3)
                    style = args.get("style", "solid")

                    if x_pos is None or not isinstance(x_pos, (int, float)):
                        continue

                    x_pos = int(x_pos)

                    if isinstance(color, list):
                        color = tuple(color)
                    elif isinstance(color, str):
                        color = (255, 0, 0)

                    tool_calls.append((x_pos, color, thickness, style))

            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        return tool_calls


class ImageActionRegistry:
    """Registry for managing image actions."""

    def __init__(self):
        self._actions: List[ImageAction] = []
        self._register_default_actions()

    def _register_default_actions(self):
        """Register default image actions."""
        self.register(ShowOriginalImageAction())
        self.register(DetectAndCropBoxesAction())
        self.register(MarkPointAction())
        self.register(DrawLineAction())
        self.register(DrawHorizontalLineAction())
        self.register(DrawVerticalLineAction())

    def register(self, action: ImageAction):
        """Register a new image action."""
        self._actions.append(action)

    def get_applicable_actions(self, image_input: Union[str, Image.Image], response_text: str, **kwargs) -> List[ImageAction]:
        """Get all applicable actions for the given inputs."""
        return [action for action in self._actions if action.is_applicable(image_input, response_text, **kwargs)]

    def get_all_actions(self) -> List[ImageAction]:
        """Get all registered actions."""
        return self._actions.copy()


# Global registry instance
_global_registry = None


def _get_global_registry() -> ImageActionRegistry:
    """Get the global registry instance, creating it if it doesn't exist."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ImageActionRegistry()
    return _global_registry


def get_image_action_registry() -> ImageActionRegistry:
    """Public function to get the image action registry."""
    return _get_global_registry()


def display_image_with_actions(image_path: str, response_text: str = "", **kwargs):
    """Display image using all applicable actions with chaining support.

    Args:
        image_path: Path to the image file
        response_text: The response text that might contain relevant information
        **kwargs: Additional arguments passed to actions
    """
    if not image_path:
        st.info('No image path provided.')
        return

    if not os.path.exists(image_path):
        st.info(f'Image not found: {image_path}')
        return

    # Use global registry instance
    registry = _get_global_registry()

    # Get all applicable actions
    applicable_actions = registry.get_applicable_actions(image_path, response_text, **kwargs)

    if not applicable_actions:
        st.info('No applicable image actions found.')
        return

        # Create action selection and chaining controls
    with st.expander("🎨 Image Display Options", expanded=True):
        selected_action_names = st.multiselect(
            "Select image actions to apply:",
            options=[action.name for action in applicable_actions],
            default=[action.name for action in applicable_actions],
            help="Choose which image processing actions to apply"
        )

        chain_actions = st.checkbox(
            "Chain actions",
            value=False,
            help="Apply each action to the result of the previous action instead of the original image"
        )

    selected_actions = [action for action in applicable_actions if action.name in selected_action_names]

    if not selected_actions:
        return

    # Execute actions
    current_input = image_path  # Start with original image path

    for i, action in enumerate(selected_actions):
        try:
            with st.container():
                if len(selected_actions) > 1:
                    st.markdown(f"**{action.name}**")

                # Execute action and get results
                results = action.execute(current_input, response_text, **kwargs)

                # Display the results
                if results:
                    action._render_image_grid(results)

                    # If chaining is enabled, use the first result as input for the next action
                    # print(f"action: {action.name}, {chain_actions=}, {results=}")
                    if not isinstance(action, DetectAndCropBoxesAction) and chain_actions and results:
                        current_input = results[-1][0]  # Use first result image
                else:
                    st.info(f"No results from {action.name}")

        except Exception as e:
            st.error(f"Error executing {action.name}: {e}")