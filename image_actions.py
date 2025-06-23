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
from typing import List, Optional, Tuple, Any

import streamlit as st
from PIL import Image

# Import functions from the actions toolbox
import sys
import importlib.util
verl_spec = importlib.util.find_spec("verl")
verl_path = verl_spec.submodule_search_locations[0]
actions_path = os.path.join(verl_path, "toolbox")
sys.path.append(actions_path)
from actions import execute_zoom_in_tool, execute_mark_points_tool, execute_draw_line_tool


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
    def execute(self, image_path: str, response_text: str, **kwargs) -> None:
        """Execute the image action.

        Args:
            image_path: Path to the image file
            response_text: The response text that might contain relevant information
            **kwargs: Additional arguments that might be needed
        """
        pass

    @abstractmethod
    def is_applicable(self, image_path: str, response_text: str, **kwargs) -> bool:
        """Check if this action is applicable for the given inputs.

        Args:
            image_path: Path to the image file
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

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Safely load an image file.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object or None if loading fails
        """
        try:
            return Image.open(image_path)
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

    def execute(self, image_path: str, response_text: str, **kwargs) -> None:
        """Display the original image."""
        image = self._load_image(image_path)
        if image:
            self._render_image_grid([(image, f"Image: {os.path.basename(image_path)}")], cols=1)

    def is_applicable(self, image_path: str, response_text: str, **kwargs) -> bool:
        """Always applicable if image path exists."""
        return bool(image_path and os.path.exists(image_path))


class DetectAndCropBoxesAction(ImageAction):
    """Action to detect and crop boxes from image_zoom_in_tool calls using actions.py."""
    tool_name = ["image_zoom_in_tool"]

    @property
    def name(self) -> str:
        return "Detect and Crop Boxes"

    @property
    def description(self) -> str:
        return "Extract box coordinates from image_zoom_in_tool calls and display cropped regions"

    def execute(self, image_path: str, response_text: str, **kwargs) -> None:
        """Extract tool calls and display cropped regions."""
        image = self._load_image(image_path)
        if not image:
            return

        tool_calls = self._extract_zoom_in_tool_calls(response_text)

        if not tool_calls:
            return

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

        self._render_image_grid(cropped_images)

    def is_applicable(self, image_path: str, response_text: str, **kwargs) -> bool:
        """Applicable if image exists and response contains image_zoom_in_tool calls."""
        if not (image_path and os.path.exists(image_path) and response_text):
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

    def execute(self, image_path: str, response_text: str, **kwargs) -> None:
        """Extract tool calls and display images with marked points."""
        image = self._load_image(image_path)
        if not image:
            return

        tool_calls = self._extract_mark_points_tool_calls(response_text)

        if not tool_calls:
            return

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

        self._render_image_grid(marked_images)

    def is_applicable(self, image_path: str, response_text: str, **kwargs) -> bool:
        """Applicable if image exists and response contains image_mark_points_tool calls."""
        if not (image_path and os.path.exists(image_path) and response_text):
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

    def execute(self, image_path: str, response_text: str, **kwargs) -> None:
        """Extract tool calls and display images with drawn lines."""
        image = self._load_image(image_path)
        if not image:
            return

        tool_calls = self._extract_draw_line_tool_calls(response_text)

        if not tool_calls:
            return

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

        self._render_image_grid(line_images)

    def is_applicable(self, image_path: str, response_text: str, **kwargs) -> bool:
        """Applicable if image exists and response contains draw_line_tool calls."""
        if not (image_path and os.path.exists(image_path) and response_text):
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

    def register(self, action: ImageAction):
        """Register a new image action."""
        self._actions.append(action)

    def get_applicable_actions(self, image_path: str, response_text: str, **kwargs) -> List[ImageAction]:
        """Get all applicable actions for the given inputs."""
        return [action for action in self._actions if action.is_applicable(image_path, response_text, **kwargs)]

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
    """Display image using all applicable actions.

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

    # Create action selection if multiple actions are available
    if len(applicable_actions) > 1:
        with st.expander("🎨 Image Display Options", expanded=True):
            selected_action_names = st.multiselect(
                "Select image actions to apply:",
                options=[action.name for action in applicable_actions],
                default=[action.name for action in applicable_actions],
                help="Choose which image processing actions to apply"
            )
            selected_actions = [action for action in applicable_actions if action.name in selected_action_names]
    else:
        selected_actions = applicable_actions

    # Execute selected actions
    for action in selected_actions:
        try:
            with st.container():
                if len(selected_actions) > 1:
                    st.markdown(f"**{action.name}**")
                action.execute(image_path, response_text, **kwargs)
        except Exception as e:
            st.error(f"Error executing {action.name}: {e}")