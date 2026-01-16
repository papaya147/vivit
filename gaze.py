from typing import List, Tuple

import einops
import matplotlib.pyplot as plt
import numpy as np


def gaussian_mask(
    gaze: Tuple[float, float], sigma: float, shape: Tuple[int, int]
) -> np.ndarray:
    """
    Generates a single 2D Gaussian mask for a specific gaze point.

    Args:
        gaze: (x, y) percentages of the center.
        sigma: Standard deviation (spread) of the Gaussian.
        shape: (height, width) of the output mask.

    Returns:
        np.ndarray: A 2D array of the specified shape.
    """
    h, w = shape
    x0, y0 = gaze
    x0 = x0 * h
    y0 = y0 * w

    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    x, y = np.meshgrid(x, y)

    dist_sq = (x - x0) ** 2 + (y - y0) ** 2
    mask = np.exp(-dist_sq / (2.0 * sigma**2))

    return mask


def decaying_gaussian_mask(
    gaze_points: List[Tuple[float, float]],
    sigma: float,
    shape: Tuple[int, int],
    decay: float = 0.9,
) -> np.ndarray:
    """
    Generates a cumulative heatmap where older points fade away.

    Args:
        gaze_points: List of (x, y) tuples over time.
        sigma: Spread of the gaussian.
        shape: (height, width) of the image.
        decay: Rate at which previous points fade (0.0 to 1.0).
               Lower = faster fade (shorter trail).
               Higher = slower fade (longer trail).
    """
    heatmap = np.zeros(shape, dtype=np.float32)

    for point in gaze_points:
        heatmap = heatmap * decay

        if point is None:
            continue

        current_mask = gaussian_mask(point, sigma, shape)
        heatmap = np.maximum(heatmap, current_mask)

    return heatmap


def patchify(gaze_mask: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        gaze_mask: Shape (B, H, W) or (H, W).
        patch_size: (H, W) of each patch in pixels.

    Returns:
        np.ndarray: Shape (B, Grid_Rows, Grid_Cols, Patch_H, Patch_W)
                    Example: (1, 14, 14, 6, 6)
    """
    y_patch_size, x_patch_size = patch_size

    if len(gaze_mask.shape) == 2:
        gaze_mask = gaze_mask[np.newaxis, :, :]

    B, H, W = gaze_mask.shape

    assert H % y_patch_size == 0 and W % x_patch_size == 0, (
        f"Image size ({H},{W}) must be divisible by patch size ({y_patch_size},{x_patch_size})"
    )

    y_patches = H // y_patch_size
    x_patches = W // x_patch_size

    # Reshape to break H and W into (grid_coord, pixel_coord)
    # Shape: (B, Grid_Y, Pixel_Y, Grid_X, Pixel_X)
    gaze_mask = gaze_mask.reshape(B, y_patches, y_patch_size, x_patches, x_patch_size)

    # Transpose to group the grid coordinates together
    # Shape: (B, Grid_Y, Grid_X, Pixel_Y, Pixel_X)
    gaze_mask = gaze_mask.transpose(0, 1, 3, 2, 4)

    return gaze_mask


def plot_patches(patches: np.ndarray, spacing: int = 1):
    """
    Visualizes patches.
    Requires NO extra arguments because layout is preserved in the tensor shape.

    Args:
        patches: Shape (B, Grid_Rows, Grid_Cols, Ph, Pw)
                 or    (Grid_Rows, Grid_Cols, Ph, Pw).
        spacing: Spacing between patches in pixels.
    """
    if len(patches.shape) == 5:
        patches = patches[0]

    # Unpack dimensions directly from the tensor
    # Shape: (Grid_Rows, Grid_Cols, Patch_H, Patch_W)
    rows, cols, ph, pw = patches.shape

    full_h = (rows * ph) + ((rows - 1) * spacing)
    full_w = (cols * pw) + ((cols - 1) * spacing)

    canvas = np.full((full_h, full_w), np.nan)

    for i in range(rows):
        for j in range(cols):
            y_start = i * (ph + spacing)
            x_start = j * (pw + spacing)

            # Extract the specific patch
            patch = patches[i, j]

            canvas[y_start : y_start + ph, x_start : x_start + pw] = patch

    plt.figure(figsize=(8, 8))
    cmap = plt.cm.hot
    cmap.set_bad(color="grey")
    plt.imshow(canvas, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Layout: {rows}x{cols} | Patch Size: {ph}x{pw}")
    plt.show()


mask = decaying_gaussian_mask([(0.5, 0.5), (0.4, 0.6), (1, 0.5)], 5, (84, 84), 0.5)

patched_mask = patchify(mask, patch_size=(12, 6))
plot_patches(patched_mask, spacing=1)
