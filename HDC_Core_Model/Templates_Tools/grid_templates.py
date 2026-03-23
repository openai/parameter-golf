"""
Grid-Level Template Functions with Dynamic Composition

This module provides actual grid transformation functions that operate on
pixel data, not HDC vectors. These are used to:
1. Actually transform grids (rotate, flip, color_swap, gravity, etc.)
2. Find which transformation explains input→output
3. Compose multiple transformations in sequence

Integration with HDC:
- GridTemplateEngine uses HDC for template selection/matching
- Encodes transformed grids to compare with target output
- Stores successful transformation recipes for future use

Example:
    >>> engine = GridTemplateEngine(hdc, encoder)
    >>> # Find what transforms input to output
    >>> recipe = engine.discover_transformation(input_grid, output_grid)
    >>> # Apply to new input
    >>> result = engine.apply_recipe(new_input, recipe)
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy


# Type alias for grids
Grid = List[List[int]]


# =============================================================================
# ATOMIC GRID TRANSFORMATIONS
# =============================================================================

def rotate_90(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    if not grid or not grid[0]:
        return grid
    return [list(row) for row in zip(*grid[::-1])]


def rotate_180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    if not grid:
        return grid
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: Grid) -> Grid:
    """Rotate grid 270 degrees clockwise (90 counter-clockwise)."""
    if not grid or not grid[0]:
        return grid
    return [list(row)[::-1] for row in zip(*grid)]


def flip_horizontal(grid: Grid) -> Grid:
    """Flip grid horizontally (mirror left-right)."""
    if not grid:
        return grid
    return [row[::-1] for row in grid]


def flip_vertical(grid: Grid) -> Grid:
    """Flip grid vertically (mirror top-bottom)."""
    if not grid:
        return grid
    return grid[::-1]


def flip_diagonal(grid: Grid) -> Grid:
    """Flip grid along main diagonal (transpose)."""
    if not grid or not grid[0]:
        return grid
    return [list(row) for row in zip(*grid)]


def flip_antidiagonal(grid: Grid) -> Grid:
    """Flip grid along anti-diagonal."""
    if not grid or not grid[0]:
        return grid
    return [list(row) for row in zip(*[r[::-1] for r in grid[::-1]])]


def identity(grid: Grid) -> Grid:
    """Return grid unchanged."""
    return deepcopy(grid)


# =============================================================================
# COLOR TRANSFORMATIONS
# =============================================================================

def color_swap(grid: Grid, color1: int, color2: int) -> Grid:
    """Swap two colors in the grid."""
    result = deepcopy(grid)
    for y in range(len(result)):
        for x in range(len(result[0])):
            if result[y][x] == color1:
                result[y][x] = color2
            elif result[y][x] == color2:
                result[y][x] = color1
    return result


def color_replace(grid: Grid, old_color: int, new_color: int) -> Grid:
    """Replace all occurrences of one color with another."""
    result = deepcopy(grid)
    for y in range(len(result)):
        for x in range(len(result[0])):
            if result[y][x] == old_color:
                result[y][x] = new_color
    return result


def color_invert(grid: Grid, max_color: int = 9) -> Grid:
    """Invert colors (0→max, max→0)."""
    result = deepcopy(grid)
    for y in range(len(result)):
        for x in range(len(result[0])):
            result[y][x] = max_color - result[y][x]
    return result


def fill_background(grid: Grid, color: int) -> Grid:
    """Fill background (0s) with a color."""
    result = deepcopy(grid)
    for y in range(len(result)):
        for x in range(len(result[0])):
            if result[y][x] == 0:
                result[y][x] = color
    return result


# =============================================================================
# GRAVITY TRANSFORMATIONS
# =============================================================================

def gravity_down(grid: Grid) -> Grid:
    """Apply gravity - non-zero cells fall to bottom."""
    if not grid or not grid[0]:
        return grid
    height, width = len(grid), len(grid[0])
    result = [[0] * width for _ in range(height)]
    
    for x in range(width):
        # Collect non-zero cells in column
        column = [grid[y][x] for y in range(height) if grid[y][x] != 0]
        # Place at bottom
        for i, val in enumerate(reversed(column)):
            result[height - 1 - i][x] = val
    
    return result


def gravity_up(grid: Grid) -> Grid:
    """Apply gravity upward - non-zero cells rise to top."""
    if not grid or not grid[0]:
        return grid
    height, width = len(grid), len(grid[0])
    result = [[0] * width for _ in range(height)]
    
    for x in range(width):
        # Collect non-zero cells in column
        column = [grid[y][x] for y in range(height) if grid[y][x] != 0]
        # Place at top
        for i, val in enumerate(column):
            result[i][x] = val
    
    return result


def gravity_left(grid: Grid) -> Grid:
    """Apply gravity leftward - non-zero cells move to left."""
    if not grid or not grid[0]:
        return grid
    height, width = len(grid), len(grid[0])
    result = [[0] * width for _ in range(height)]
    
    for y in range(height):
        # Collect non-zero cells in row
        row = [grid[y][x] for x in range(width) if grid[y][x] != 0]
        # Place at left
        for i, val in enumerate(row):
            result[y][i] = val
    
    return result


def gravity_right(grid: Grid) -> Grid:
    """Apply gravity rightward - non-zero cells move to right."""
    if not grid or not grid[0]:
        return grid
    height, width = len(grid), len(grid[0])
    result = [[0] * width for _ in range(height)]
    
    for y in range(height):
        # Collect non-zero cells in row
        row = [grid[y][x] for x in range(width) if grid[y][x] != 0]
        # Place at right
        for i, val in enumerate(reversed(row)):
            result[y][width - 1 - i] = val
    
    return result


# =============================================================================
# STRUCTURAL TRANSFORMATIONS
# =============================================================================

def tile_2x2(grid: Grid) -> Grid:
    """Tile grid 2x2 (repeat 4 times)."""
    if not grid or not grid[0]:
        return grid
    height, width = len(grid), len(grid[0])
    result = [[0] * (width * 2) for _ in range(height * 2)]
    
    for dy in range(2):
        for dx in range(2):
            for y in range(height):
                for x in range(width):
                    result[y + dy * height][x + dx * width] = grid[y][x]
    
    return result


def tile_horizontal(grid: Grid) -> Grid:
    """Tile grid horizontally (2x wide)."""
    if not grid:
        return grid
    return [row + row for row in grid]


def tile_vertical(grid: Grid) -> Grid:
    """Tile grid vertically (2x tall)."""
    if not grid:
        return grid
    return grid + deepcopy(grid)


def scale_2x(grid: Grid) -> Grid:
    """Scale grid 2x (each cell becomes 2x2)."""
    if not grid or not grid[0]:
        return grid
    height, width = len(grid), len(grid[0])
    result = [[0] * (width * 2) for _ in range(height * 2)]
    
    for y in range(height):
        for x in range(width):
            val = grid[y][x]
            result[y * 2][x * 2] = val
            result[y * 2][x * 2 + 1] = val
            result[y * 2 + 1][x * 2] = val
            result[y * 2 + 1][x * 2 + 1] = val
    
    return result


def crop_nonzero(grid: Grid) -> Grid:
    """Crop to smallest bounding box containing non-zero cells."""
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    
    # Find bounds
    min_x, max_x = width, 0
    min_y, max_y = height, 0
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    
    if min_x > max_x:
        return [[]]
    
    return [row[min_x:max_x + 1] for row in grid[min_y:max_y + 1]]


def extract_color(grid: Grid, color: int) -> Grid:
    """Extract only cells of a specific color."""
    result = deepcopy(grid)
    for y in range(len(result)):
        for x in range(len(result[0])):
            if result[y][x] != color:
                result[y][x] = 0
    return result


def fill_enclosed(grid: Grid, fill_color: int = 1) -> Grid:
    """Fill enclosed regions (background cells surrounded by non-zero)."""
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    result = deepcopy(grid)
    visited = [[False] * width for _ in range(height)]
    
    # Flood fill from edges to find all cells connected to border
    def flood_fill_border(y: int, x: int):
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        if visited[y][x] or grid[y][x] != 0:
            return
        visited[y][x] = True
        flood_fill_border(y - 1, x)
        flood_fill_border(y + 1, x)
        flood_fill_border(y, x - 1)
        flood_fill_border(y, x + 1)
    
    # Start from all border cells
    for x in range(width):
        if not visited[0][x] and grid[0][x] == 0:
            flood_fill_border(0, x)
        if not visited[height-1][x] and grid[height-1][x] == 0:
            flood_fill_border(height - 1, x)
    for y in range(height):
        if not visited[y][0] and grid[y][0] == 0:
            flood_fill_border(y, 0)
        if not visited[y][width-1] and grid[y][width-1] == 0:
            flood_fill_border(y, width - 1)
    
    # Fill unvisited zero cells
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 0 and not visited[y][x]:
                result[y][x] = fill_color
    
    return result


def outline(grid: Grid, outline_color: int = 1) -> Grid:
    """Create outline around non-zero regions (adds to background)."""
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    result = deepcopy(grid)
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 0:
                # Check if adjacent to non-zero
                neighbors = []
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbors.append(grid[ny][nx])
                if any(n != 0 for n in neighbors):
                    result[y][x] = outline_color
    
    return result


# =============================================================================
# MORPHOLOGICAL OUTLINE TRANSFORMATIONS (for MORPH_OUTLINE tasks)
# =============================================================================

def mark_boundary(grid: Grid, boundary_color: int = 2, interior_color: int = None) -> Grid:
    """
    Mark boundary cells of a filled shape with boundary_color.
    
    A boundary cell is a non-zero cell that:
    1. Is on the edge of the grid, OR
    2. Is adjacent to a zero (background) cell
    
    This is the key operation for MORPH_OUTLINE tasks.
    
    Args:
        grid: Input grid
        boundary_color: Color to use for boundary cells
        interior_color: If specified, use this for interior cells (else keep original)
    
    Example:
        Input:  [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        Output: [[2, 2, 2], [2, 1, 2], [2, 2, 2]] (with boundary_color=2)
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    result = deepcopy(grid)
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:  # Only process non-zero cells
                is_boundary = False
                
                # Check if on edge of grid
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    is_boundary = True
                else:
                    # Check if adjacent to zero (4-connected)
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if grid[ny][nx] == 0:
                                is_boundary = True
                                break
                
                if is_boundary:
                    result[y][x] = boundary_color
                elif interior_color is not None:
                    result[y][x] = interior_color
    
    return result


def mark_boundary_8connected(grid: Grid, boundary_color: int = 2, interior_color: int = None) -> Grid:
    """
    Mark boundary cells using 8-connectivity (includes diagonals).
    
    A cell is boundary if it's on the grid edge or has any diagonal neighbor that's zero.
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    result = deepcopy(grid)
    
    # 8-connected neighbors (including diagonals)
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                is_boundary = False
                
                # Check if on edge
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    is_boundary = True
                else:
                    # Check 8-connected neighbors
                    for dy, dx in neighbors_8:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if grid[ny][nx] == 0:
                                is_boundary = True
                                break
                
                if is_boundary:
                    result[y][x] = boundary_color
                elif interior_color is not None:
                    result[y][x] = interior_color
    
    return result


def extract_boundary(grid: Grid) -> Grid:
    """
    Extract only the boundary cells, removing interior (set to 0).
    
    This creates a hollow outline from a filled shape.
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    result = [[0] * width for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                is_boundary = False
                
                # Check if on edge
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    is_boundary = True
                else:
                    # Check 4-connected neighbors
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if grid[ny][nx] == 0:
                                is_boundary = True
                                break
                
                if is_boundary:
                    result[y][x] = grid[y][x]  # Keep original color
    
    return result


def extract_interior(grid: Grid) -> Grid:
    """
    Extract only interior cells (non-boundary), setting boundary to 0.
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    result = [[0] * width for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                is_interior = True
                
                # Check if on edge (not interior)
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    is_interior = False
                else:
                    # Check 4-connected neighbors
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if grid[ny][nx] == 0:
                                is_interior = False
                                break
                
                if is_interior:
                    result[y][x] = grid[y][x]
    
    return result


def dilate(grid: Grid, iterations: int = 1, fill_color: int = None) -> Grid:
    """
    Dilate/grow shapes by expanding into adjacent background cells.
    
    Args:
        grid: Input grid
        iterations: Number of dilation iterations
        fill_color: Color to use for new cells (None = copy from neighbor)
    """
    if not grid or not grid[0]:
        return grid
    
    result = deepcopy(grid)
    height, width = len(grid), len(grid[0])
    
    for _ in range(iterations):
        new_result = deepcopy(result)
        
        for y in range(height):
            for x in range(width):
                if result[y][x] == 0:
                    # Check if adjacent to non-zero
                    neighbor_colors = []
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if result[ny][nx] != 0:
                                neighbor_colors.append(result[ny][nx])
                    
                    if neighbor_colors:
                        if fill_color is not None:
                            new_result[y][x] = fill_color
                        else:
                            # Use most common neighbor color
                            new_result[y][x] = max(set(neighbor_colors), key=neighbor_colors.count)
        
        result = new_result
    
    return result


def erode(grid: Grid, iterations: int = 1) -> Grid:
    """
    Erode/shrink shapes by removing boundary cells.
    
    Opposite of dilate - removes the outer layer of shapes.
    """
    if not grid or not grid[0]:
        return grid
    
    result = deepcopy(grid)
    height, width = len(grid), len(grid[0])
    
    for _ in range(iterations):
        new_result = deepcopy(result)
        
        for y in range(height):
            for x in range(width):
                if result[y][x] != 0:
                    # Check if on boundary
                    is_boundary = False
                    
                    if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                        is_boundary = True
                    else:
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if result[ny][nx] == 0:
                                    is_boundary = True
                                    break
                    
                    if is_boundary:
                        new_result[y][x] = 0
        
        result = new_result
    
    return result


def morph_close(grid: Grid, iterations: int = 1) -> Grid:
    """
    Morphological close: dilate then erode.
    
    This operation fills small holes and gaps in shapes.
    Useful for closing small internal holes while preserving overall shape.
    """
    result = dilate(grid, iterations)
    result = erode(result, iterations)
    return result


def morph_open(grid: Grid, iterations: int = 1) -> Grid:
    """
    Morphological open: erode then dilate.
    
    This operation removes small noise/protrusions from shapes.
    Useful for removing small isolated pixels while preserving larger shapes.
    """
    result = erode(grid, iterations)
    result = dilate(result, iterations)
    return result


def fill_holes(grid: Grid, fill_color: int = None) -> Grid:
    """
    Fill all internal holes in shapes.
    
    This is similar to fill_enclosed but infers the fill color from
    the surrounding shape if not specified.
    
    Args:
        fill_color: Color to fill holes with. If None, uses dominant color of surrounding shape.
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    
    # If fill_color not specified, find dominant non-zero color
    if fill_color is None:
        color_counts = {}
        for y in range(height):
            for x in range(width):
                if grid[y][x] != 0:
                    color_counts[grid[y][x]] = color_counts.get(grid[y][x], 0) + 1
        if color_counts:
            fill_color = max(color_counts, key=color_counts.get)
        else:
            fill_color = 1
    
    return fill_enclosed(grid, fill_color)


def morph_outline_recolor(grid: Grid, boundary_color: int = 2) -> Grid:
    """
    MORPH_OUTLINE specific: Recolor boundary cells while keeping interior unchanged.
    
    This is the exact operation needed for MORPH_OUTLINE tasks.
    Detects the dominant non-zero color and uses it to identify the shape,
    then recolors only boundary cells.
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    
    # Find non-zero colors
    color_counts = {}
    for row in grid:
        for cell in row:
            if cell != 0:
                color_counts[cell] = color_counts.get(cell, 0) + 1
    
    if not color_counts:
        return deepcopy(grid)
    
    # Get dominant color (the filled shape color)
    dominant_color = max(color_counts, key=color_counts.get)
    
    # Mark boundaries of the dominant colored shape
    return mark_boundary(grid, boundary_color=boundary_color, interior_color=dominant_color)


def detect_and_mark_boundary(grid: Grid) -> Grid:
    """
    Detect filled shapes and mark their boundaries with a new color.
    
    Automatically chooses boundary color based on what's in the grid:
    - If grid has color 1, boundary becomes 2
    - If grid has colors 1-n, boundary becomes n+1 (capped at 9)
    """
    if not grid or not grid[0]:
        return grid
    
    # Find max color in grid
    max_color = 0
    for row in grid:
        for cell in row:
            if cell > max_color:
                max_color = cell
    
    # Boundary color is one higher (capped at 9)
    boundary_color = min(max_color + 1, 9) if max_color > 0 else 1
    
    return mark_boundary(grid, boundary_color=boundary_color)


# =============================================================================
# DRAWING PRIMITIVES (Lines, Connections, etc.)
# =============================================================================

def draw_line(grid: Grid, start: Tuple[int, int], end: Tuple[int, int], color: int) -> Grid:
    """Draw a line between two points using Bresenham's algorithm."""
    if not grid or not grid[0]:
        return grid
    
    result = deepcopy(grid)
    height, width = len(grid), len(grid[0])
    
    y0, x0 = start
    y1, x1 = end
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        if 0 <= y0 < height and 0 <= x0 < width:
            result[y0][x0] = color
        
        if y0 == y1 and x0 == x1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
    return result


def connect_points(grid: Grid, color: int = None) -> Grid:
    """
    Connect points of the same color with lines.
    
    If color is None, connects points for all colors present.
    Only connects if there are exactly 2 points of that color.
    """
    if not grid or not grid[0]:
        return grid
    
    result = deepcopy(grid)
    height, width = len(grid), len(grid[0])
    
    # Find points by color
    points_by_color = {}
    for y in range(height):
        for x in range(width):
            c = grid[y][x]
            if c != 0:
                if c not in points_by_color:
                    points_by_color[c] = []
                points_by_color[c].append((y, x))
    
    # Connect points
    colors_to_process = [color] if color is not None else points_by_color.keys()
    
    for c in colors_to_process:
        if c in points_by_color and len(points_by_color[c]) == 2:
            start = points_by_color[c][0]
            end = points_by_color[c][1]
            result = draw_line(result, start, end, c)
            
    return result


def connect_all_pairs(grid: Grid) -> Grid:
    """Connect all pairs of same-colored points."""
    return connect_points(grid, color=None)


def draw_rectangle(grid: Grid, start: Tuple[int, int], end: Tuple[int, int], color: int, fill: bool = False) -> Grid:
    """Draw a rectangle defined by two corners."""
    if not grid or not grid[0]:
        return grid
    
    result = deepcopy(grid)
    height, width = len(grid), len(grid[0])
    
    y0, x0 = start
    y1, x1 = end
    
    min_y, max_y = min(y0, y1), max(y0, y1)
    min_x, max_x = min(x0, x1), max(x0, x1)
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if 0 <= y < height and 0 <= x < width:
                if fill:
                    result[y][x] = color
                else:
                    # Outline only
                    if y == min_y or y == max_y or x == min_x or x == max_x:
                        result[y][x] = color
    
    return result


def draw_box_around_object(grid: Grid, object_color: int, box_color: int) -> Grid:
    """Draw a bounding box around all cells of a specific color."""
    if not grid or not grid[0]:
        return grid
    
    # Find bounding box
    min_y, max_y = len(grid), 0
    min_x, max_x = len(grid[0]), 0
    found = False
    
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] == object_color:
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                found = True
    
    if not found:
        return deepcopy(grid)
    
    # Expand box by 1 cell if possible
    min_y = max(0, min_y - 1)
    max_y = min(len(grid) - 1, max_y + 1)
    min_x = max(0, min_x - 1)
    max_x = min(len(grid[0]) - 1, max_x + 1)
    
    return draw_rectangle(grid, (min_y, min_x), (max_y, max_x), box_color, fill=False)


def flood_fill(grid: Grid, start: Tuple[int, int], color: int) -> Grid:
    """Flood fill from a start point."""
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    y, x = start
    
    if not (0 <= y < height and 0 <= x < width):
        return grid
    
    target_color = grid[y][x]
    if target_color == color:
        return grid
    
    result = deepcopy(grid)
    stack = [(y, x)]
    
    while stack:
        cy, cx = stack.pop()
        if result[cy][cx] == target_color:
            result[cy][cx] = color
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    stack.append((ny, nx))
    
    return result


# =============================================================================
# OBJECT TRANSLATION TRANSFORMATIONS (for OBJ_TRANS tasks)
# =============================================================================

def find_objects(grid: Grid) -> List[Dict]:
    """
    Find all connected components (objects) in the grid.
    
    Returns list of objects, each with:
    - 'cells': List of (y, x) coordinates
    - 'color': The object's color
    - 'bbox': (min_y, min_x, max_y, max_x) bounding box
    - 'center': (center_y, center_x) centroid
    """
    if not grid or not grid[0]:
        return []
    
    height, width = len(grid), len(grid[0])
    visited = [[False] * width for _ in range(height)]
    objects = []
    
    def flood_fill(start_y: int, start_x: int, color: int) -> List[Tuple[int, int]]:
        """Find all cells connected to (start_y, start_x) with same color."""
        cells = []
        stack = [(start_y, start_x)]
        
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= height or x < 0 or x >= width:
                continue
            if visited[y][x] or grid[y][x] != color:
                continue
            
            visited[y][x] = True
            cells.append((y, x))
            
            # 4-connected neighbors
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
        
        return cells
    
    for y in range(height):
        for x in range(width):
            if not visited[y][x] and grid[y][x] != 0:
                color = grid[y][x]
                cells = flood_fill(y, x, color)
                
                if cells:
                    # Calculate bounding box
                    ys = [c[0] for c in cells]
                    xs = [c[1] for c in cells]
                    min_y, max_y = min(ys), max(ys)
                    min_x, max_x = min(xs), max(xs)
                    
                    # Calculate centroid
                    center_y = sum(ys) / len(cells)
                    center_x = sum(xs) / len(cells)
                    
                    objects.append({
                        'cells': cells,
                        'color': color,
                        'bbox': (min_y, min_x, max_y, max_x),
                        'center': (center_y, center_x)
                    })
    
    return objects


def translate_grid(grid: Grid, dy: int, dx: int, wrap: bool = False) -> Grid:
    """
    Translate entire grid contents by (dy, dx).
    
    Args:
        grid: Input grid
        dy: Vertical shift (positive = down)
        dx: Horizontal shift (positive = right)
        wrap: If True, wrap around edges; if False, cells shift out are lost
    
    Returns:
        Translated grid
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    result = [[0] * width for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                new_y = y + dy
                new_x = x + dx
                
                if wrap:
                    new_y = new_y % height
                    new_x = new_x % width
                
                if 0 <= new_y < height and 0 <= new_x < width:
                    result[new_y][new_x] = grid[y][x]
    
    return result


def translate_object_by_delta(grid: Grid, dy: int, dx: int) -> Grid:
    """
    Translate all non-zero cells by (dy, dx).
    
    This is the core operation for OBJ_TRANS tasks.
    """
    return translate_grid(grid, dy, dx, wrap=False)


def translate_up(grid: Grid, amount: int = 1) -> Grid:
    """Move all objects up by `amount` cells."""
    return translate_grid(grid, -amount, 0)


def translate_down(grid: Grid, amount: int = 1) -> Grid:
    """Move all objects down by `amount` cells."""
    return translate_grid(grid, amount, 0)


def translate_left(grid: Grid, amount: int = 1) -> Grid:
    """Move all objects left by `amount` cells."""
    return translate_grid(grid, 0, -amount)


def translate_right(grid: Grid, amount: int = 1) -> Grid:
    """Move all objects right by `amount` cells."""
    return translate_grid(grid, 0, amount)


def translate_up_left(grid: Grid, amount: int = 1) -> Grid:
    """Move all objects diagonally up-left."""
    return translate_grid(grid, -amount, -amount)


def translate_up_right(grid: Grid, amount: int = 1) -> Grid:
    """Move all objects diagonally up-right."""
    return translate_grid(grid, -amount, amount)


def translate_down_left(grid: Grid, amount: int = 1) -> Grid:
    """Move all objects diagonally down-left."""
    return translate_grid(grid, amount, -amount)


def translate_down_right(grid: Grid, amount: int = 1) -> Grid:
    """Move all objects diagonally down-right."""
    return translate_grid(grid, amount, amount)


def detect_translation(input_grid: Grid, output_grid: Grid) -> Optional[Tuple[int, int]]:
    """
    Detect the translation vector (dy, dx) that transforms input to output.
    
    Compares centroids of objects between input and output.
    
    Returns:
        (dy, dx) translation vector, or None if detection fails
    """
    if not input_grid or not output_grid:
        return None
    
    # Find objects in both grids
    input_objects = find_objects(input_grid)
    output_objects = find_objects(output_grid)
    
    if not input_objects or not output_objects:
        return None
    
    # Match objects by color and size
    translations = []
    
    for in_obj in input_objects:
        for out_obj in output_objects:
            if in_obj['color'] == out_obj['color'] and len(in_obj['cells']) == len(out_obj['cells']):
                # Calculate translation
                dy = out_obj['center'][0] - in_obj['center'][0]
                dx = out_obj['center'][1] - in_obj['center'][1]
                translations.append((round(dy), round(dx)))
    
    if not translations:
        return None
    
    # Return most common translation (majority vote)
    from collections import Counter
    most_common = Counter(translations).most_common(1)[0][0]
    return most_common


def translate_with_inference(input_grid: Grid, output_sample: Grid) -> Grid:
    """
    Apply translation inferred from comparing input to output_sample.
    
    This is the key function for OBJ_TRANS tasks - automatically detects
    the translation vector from the training pair.
    """
    translation = detect_translation(input_grid, output_sample)
    
    if translation:
        dy, dx = translation
        return translate_grid(input_grid, dy, dx)
    
    return deepcopy(input_grid)


def center_object(grid: Grid) -> Grid:
    """
    Move object(s) to the center of the grid.
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    
    # Find bounding box of all non-zero cells
    min_y, min_x = height, width
    max_y, max_x = 0, 0
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
    
    if min_y > max_y:  # No non-zero cells
        return deepcopy(grid)
    
    # Calculate center of object
    obj_center_y = (min_y + max_y) / 2
    obj_center_x = (min_x + max_x) / 2
    
    # Calculate center of grid
    grid_center_y = (height - 1) / 2
    grid_center_x = (width - 1) / 2
    
    # Calculate translation
    dy = round(grid_center_y - obj_center_y)
    dx = round(grid_center_x - obj_center_x)
    
    return translate_grid(grid, dy, dx)


def align_to_corner(grid: Grid, corner: str = 'top_left') -> Grid:
    """
    Align object(s) to a specific corner.
    
    Args:
        corner: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    
    # Find bounding box
    min_y, min_x = height, width
    max_y, max_x = 0, 0
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
    
    if min_y > max_y:
        return deepcopy(grid)
    
    # Calculate translation based on corner
    if corner == 'top_left':
        dy, dx = -min_y, -min_x
    elif corner == 'top_right':
        dy, dx = -min_y, (width - 1 - max_x)
    elif corner == 'bottom_left':
        dy, dx = (height - 1 - max_y), -min_x
    elif corner == 'bottom_right':
        dy, dx = (height - 1 - max_y), (width - 1 - max_x)
    else:
        return deepcopy(grid)
    
    return translate_grid(grid, dy, dx)


def align_to_edge(grid: Grid, edge: str = 'top') -> Grid:
    """
    Align object(s) to a specific edge.
    
    Args:
        edge: 'top', 'bottom', 'left', 'right'
    """
    if not grid or not grid[0]:
        return grid
    
    height, width = len(grid), len(grid[0])
    
    # Find bounding box
    min_y, min_x = height, width
    max_y, max_x = 0, 0
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
    
    if min_y > max_y:
        return deepcopy(grid)
    
    # Calculate translation based on edge
    if edge == 'top':
        dy, dx = -min_y, 0
    elif edge == 'bottom':
        dy, dx = (height - 1 - max_y), 0
    elif edge == 'left':
        dy, dx = 0, -min_x
    elif edge == 'right':
        dy, dx = 0, (width - 1 - max_x)
    else:
        return deepcopy(grid)
    
    return translate_grid(grid, dy, dx)


def morph_outline_with_color_inference(grid: Grid, output_sample: Grid = None) -> Grid:
    """
    MORPH_OUTLINE with color inference from output sample.
    
    If output_sample is provided, infer the boundary color from it.
    """
    if not grid or not grid[0]:
        return grid
    
    # Try to infer boundary color from output sample
    if output_sample:
        input_colors = set()
        output_colors = set()
        for row in grid:
            input_colors.update(c for c in row if c != 0)
        for row in output_sample:
            output_colors.update(c for c in row if c != 0)
        
        # New colors in output are likely boundary colors
        new_colors = output_colors - input_colors
        if new_colors:
            boundary_color = min(new_colors)  # Use smallest new color
            interior_color = min(input_colors) if input_colors else 1
            return mark_boundary(grid, boundary_color=boundary_color, interior_color=interior_color)
    
    # Fallback to auto-detection
    return detect_and_mark_boundary(grid)


# =============================================================================
# TRANSFORMATION RECIPE
# =============================================================================

@dataclass
class TransformationStep:
    """A single step in a transformation recipe."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.name}({param_str})"
        return self.name


@dataclass
class TransformationRecipe:
    """A recipe of transformations to apply in sequence."""
    steps: List[TransformationStep] = field(default_factory=list)
    confidence: float = 0.0
    
    def __repr__(self):
        return " → ".join(str(s) for s in self.steps)
    
    def add_step(self, name: str, **params):
        self.steps.append(TransformationStep(name, params))
        return self


# =============================================================================
# GRID TEMPLATE ENGINE
# =============================================================================

class GridTemplateEngine:
    """
    Engine for discovering and applying grid transformations.
    
    Uses HDC for:
    1. Encoding grids to compare similarity
    2. Storing/retrieving successful recipes
    
    Example:
        >>> engine = GridTemplateEngine(hdc, encoder)
        >>> recipe = engine.discover_transformation(input_grid, output_grid)
        >>> result = engine.apply_recipe(new_input, recipe)
    """
    
    # All available atomic transformations
    ATOMIC_TRANSFORMS = {
        # Geometric
        'rotate_90': (rotate_90, {}),
        'rotate_180': (rotate_180, {}),
        'rotate_270': (rotate_270, {}),
        'flip_horizontal': (flip_horizontal, {}),
        'flip_vertical': (flip_vertical, {}),
        'flip_diagonal': (flip_diagonal, {}),
        'flip_antidiagonal': (flip_antidiagonal, {}),
        'identity': (identity, {}),
        # Gravity
        'gravity_down': (gravity_down, {}),
        'gravity_up': (gravity_up, {}),
        'gravity_left': (gravity_left, {}),
        'gravity_right': (gravity_right, {}),
        # Translation (for OBJ_TRANS tasks)
        'translate_up': (translate_up, {'amount': 1}),
        'translate_down': (translate_down, {'amount': 1}),
        'translate_left': (translate_left, {'amount': 1}),
        'translate_right': (translate_right, {'amount': 1}),
        'translate_up_left': (translate_up_left, {'amount': 1}),
        'translate_up_right': (translate_up_right, {'amount': 1}),
        'translate_down_left': (translate_down_left, {'amount': 1}),
        'translate_down_right': (translate_down_right, {'amount': 1}),
        # Dynamic translation operations (for OBJ_TRANS tasks)
        'center_object': (center_object, {}),
        'translate_by_offset': (translate_object_by_delta, {'dy': 0, 'dx': 0}),
        'move_object_to': (center_object, {}),  # Alias for center
        'align_to_edge': (align_to_edge, {'edge': 'top'}),
        'align_to_center': (center_object, {}),  # Alias for center
        'shift_all_objects': (translate_object_by_delta, {'dy': 0, 'dx': 0}),
        'translate_to_corner': (align_to_corner, {'corner': 'top_left'}),
        # Structural
        'tile_2x2': (tile_2x2, {}),
        'tile_horizontal': (tile_horizontal, {}),
        'tile_vertical': (tile_vertical, {}),
        'scale_2x': (scale_2x, {}),
        'crop_nonzero': (crop_nonzero, {}),
        'outline': (outline, {'outline_color': 1}),
        # Morphological (for MORPH_OUTLINE tasks)
        'mark_boundary': (mark_boundary, {'boundary_color': 2}),
        'mark_boundary_8conn': (mark_boundary_8connected, {'boundary_color': 2}),
        'extract_boundary': (extract_boundary, {}),
        'extract_interior': (extract_interior, {}),
        'dilate': (dilate, {'iterations': 1}),
        'erode': (erode, {'iterations': 1}),
        # Fill holes operations
        'fill_enclosed': (fill_enclosed, {'fill_color': 1}),
        'fill_holes': (fill_holes, {}),
        'morph_close': (morph_close, {'iterations': 1}),
        'morph_open': (morph_open, {'iterations': 1}),
        # Outline operations
        'morph_outline_recolor': (morph_outline_recolor, {'boundary_color': 2}),
        'detect_and_mark_boundary': (detect_and_mark_boundary, {}),
        # Drawing
        'connect_points': (connect_points, {'color': 1}),
        'connect_all_pairs': (connect_all_pairs, {}),
        'draw_box_around_object': (draw_box_around_object, {'object_color': 1, 'box_color': 2}),
        'draw_rectangle': (draw_rectangle, {'start': (0, 0), 'end': (1, 1), 'color': 1, 'fill': False}),
        'flood_fill': (flood_fill, {'start': (0, 0), 'color': 1}),
    }
    
    # Transforms with color parameters
    COLOR_TRANSFORMS = [
        'color_swap', 'color_replace', 'fill_background', 'extract_color',
        'mark_boundary', 'morph_outline_recolor'  # Morphological with color params
    ]
    
    def __init__(self, hdc, encoder, decoder=None):
        """
        Initialize the engine.
        
        Args:
            hdc: SparseBinaryHDC instance
            encoder: ARCGridEncoder for encoding grids
            decoder: Optional ARCGridDecoder for decoding
        """
        self.hdc = hdc
        self.encoder = encoder
        self.decoder = decoder
        
        # Cache of (input_hash, output_hash) → recipe
        self.recipe_cache: Dict[Tuple[str, str], TransformationRecipe] = {}
    
    def _grid_hash(self, grid: Grid) -> str:
        """Create a hash string for a grid."""
        return str(grid)
    
    def _grids_match(self, a: Grid, b: Grid, tolerance: float = 0.95) -> Tuple[bool, float]:
        """Check if two grids match, return (match, accuracy)."""
        if len(a) != len(b):
            return False, 0.0
        if not a or not b:
            return a == b, 1.0 if a == b else 0.0
        if len(a[0]) != len(b[0]):
            return False, 0.0
        
        total = 0
        correct = 0
        for y in range(len(a)):
            for x in range(len(a[0])):
                total += 1
                if a[y][x] == b[y][x]:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy >= tolerance, accuracy
    
    def apply_transform(self, grid: Grid, name: str, **params) -> Grid:
        """Apply a single transformation to a grid."""
        if name in self.ATOMIC_TRANSFORMS:
            func, default_params = self.ATOMIC_TRANSFORMS[name]
            merged_params = {**default_params, **params}
            
            # Handle functions that take parameters
            if name == 'mark_boundary':
                return mark_boundary(grid,
                    boundary_color=merged_params.get('boundary_color', 2),
                    interior_color=merged_params.get('interior_color', None))
            elif name == 'mark_boundary_8conn':
                return mark_boundary_8connected(grid,
                    boundary_color=merged_params.get('boundary_color', 2),
                    interior_color=merged_params.get('interior_color', None))
            elif name == 'dilate':
                return dilate(grid,
                    iterations=merged_params.get('iterations', 1),
                    fill_color=merged_params.get('fill_color', None))
            elif name == 'erode':
                return erode(grid,
                    iterations=merged_params.get('iterations', 1))
            elif name == 'morph_outline_recolor':
                return morph_outline_recolor(grid,
                    boundary_color=merged_params.get('boundary_color', 2))
            elif name == 'outline':
                return outline(grid, outline_color=merged_params.get('outline_color', 1))
            # Translation transforms with amount parameter
            elif name in ['translate_up', 'translate_down', 'translate_left', 'translate_right',
                         'translate_up_left', 'translate_up_right', 'translate_down_left', 'translate_down_right']:
                amount = merged_params.get('amount', 1)
                return func(grid, amount=amount)
            elif name == 'translate_object':
                # Handle arbitrary (dy, dx) translation
                dy = merged_params.get('dy', 0)
                dx = merged_params.get('dx', 0)
                return translate_grid(grid, dy, dx)
            else:
                # No-parameter transforms
                return func(grid)
        
        # Color transforms with parameters
        if name == 'color_swap':
            return color_swap(grid, params.get('color1', 0), params.get('color2', 1))
        elif name == 'color_replace':
            return color_replace(grid, params.get('old_color', 0), params.get('new_color', 1))
        elif name == 'fill_background':
            return fill_background(grid, params.get('color', 1))
        elif name == 'extract_color':
            return extract_color(grid, params.get('color', 1))
        elif name == 'color_invert':
            return color_invert(grid)
        elif name == 'fill_enclosed':
            return fill_enclosed(grid, params.get('fill_color', 1))
        
        raise ValueError(f"Unknown transformation: {name}")
    
    def apply_recipe(self, grid: Grid, recipe: TransformationRecipe) -> Grid:
        """Apply a recipe of transformations to a grid."""
        result = deepcopy(grid)
        for step in recipe.steps:
            result = self.apply_transform(result, step.name, **step.params)
        return result
    
    def discover_single_transform(
        self,
        input_grid: Grid,
        output_grid: Grid,
        tolerance: float = 0.95
    ) -> Optional[TransformationRecipe]:
        """Try to find a single transformation that explains input→output."""
        
        # Try each atomic transform
        for name, (func, _) in self.ATOMIC_TRANSFORMS.items():
            if name == 'identity':
                continue  # Skip identity unless exact match
            
            try:
                result = func(input_grid)
                match, accuracy = self._grids_match(result, output_grid, tolerance)
                if match:
                    recipe = TransformationRecipe()
                    recipe.add_step(name)
                    recipe.confidence = accuracy
                    return recipe
            except Exception:
                continue
        
        # ==============================================================
        # TRANSLATION WITH INFERENCE (for OBJ_TRANS tasks)
        # This tries to detect the translation offset from input/output
        # and creates a parameterized translation recipe
        # ==============================================================
        
        # Try to detect translation offset
        translation = detect_translation(input_grid, output_grid)
        if translation:
            dy, dx = translation
            try:
                result = translate_grid(input_grid, dy, dx)
                match, accuracy = self._grids_match(result, output_grid, tolerance)
                if match:
                    recipe = TransformationRecipe()
                    recipe.add_step('translate_object', dy=dy, dx=dx)
                    recipe.confidence = accuracy
                    return recipe
            except Exception:
                pass
        
        # Try translation with various small amounts (1-5 cells)
        translation_funcs = [
            ('translate_up', translate_up),
            ('translate_down', translate_down),
            ('translate_left', translate_left),
            ('translate_right', translate_right),
            ('translate_up_left', translate_up_left),
            ('translate_up_right', translate_up_right),
            ('translate_down_left', translate_down_left),
            ('translate_down_right', translate_down_right),
        ]
        
        for amount in range(1, 6):  # Try amounts 1-5
            for name, func in translation_funcs:
                try:
                    result = func(input_grid, amount=amount)
                    match, accuracy = self._grids_match(result, output_grid, tolerance)
                    if match:
                        recipe = TransformationRecipe()
                        recipe.add_step(name, amount=amount)
                        recipe.confidence = accuracy
                        return recipe
                except Exception:
                    continue
        
        # Try color transforms
        colors_in_input = set()
        colors_in_output = set()
        for row in input_grid:
            colors_in_input.update(row)
        for row in output_grid:
            colors_in_output.update(row)
        
        all_colors = colors_in_input | colors_in_output
        new_colors = colors_in_output - colors_in_input  # Colors that appear in output but not input
        
        # ==============================================================
        # MORPHOLOGICAL TRANSFORMS WITH COLOR INFERENCE
        # This is critical for MORPH_OUTLINE tasks where we need to
        # detect boundary_color from the output
        # ==============================================================
        
        # Try mark_boundary with inferred colors
        if new_colors:
            for boundary_color in new_colors:
                # Find interior color (most common non-boundary color in output)
                interior_colors = colors_in_output - {boundary_color, 0}
                interior_color = min(interior_colors) if interior_colors else None
                
                try:
                    result = mark_boundary(input_grid, boundary_color=boundary_color, interior_color=interior_color)
                    match, accuracy = self._grids_match(result, output_grid, tolerance)
                    if match:
                        recipe = TransformationRecipe()
                        recipe.add_step('mark_boundary', boundary_color=boundary_color, interior_color=interior_color)
                        recipe.confidence = accuracy
                        return recipe
                except Exception:
                    continue
        
        # Try morph_outline_recolor with various boundary colors
        for boundary_color in range(1, 10):
            if boundary_color in colors_in_input:
                continue  # Skip if boundary color already exists
            try:
                result = morph_outline_recolor(input_grid, boundary_color=boundary_color)
                match, accuracy = self._grids_match(result, output_grid, tolerance)
                if match:
                    recipe = TransformationRecipe()
                    recipe.add_step('morph_outline_recolor', boundary_color=boundary_color)
                    recipe.confidence = accuracy
                    return recipe
            except Exception:
                continue
        
        # Try mark_boundary with 8-connectivity
        if new_colors:
            for boundary_color in new_colors:
                interior_colors = colors_in_output - {boundary_color, 0}
                interior_color = min(interior_colors) if interior_colors else None
                
                try:
                    result = mark_boundary_8connected(input_grid, boundary_color=boundary_color, interior_color=interior_color)
                    match, accuracy = self._grids_match(result, output_grid, tolerance)
                    if match:
                        recipe = TransformationRecipe()
                        recipe.add_step('mark_boundary_8conn', boundary_color=boundary_color, interior_color=interior_color)
                        recipe.confidence = accuracy
                        return recipe
                except Exception:
                    continue
        
        # Try morph_outline_with_color_inference (uses output to infer boundary color)
        try:
            result = morph_outline_with_color_inference(input_grid, output_sample=output_grid)
            match, accuracy = self._grids_match(result, output_grid, tolerance)
            if match:
                recipe = TransformationRecipe()
                # Extract inferred colors for recipe
                input_colors = set(c for row in input_grid for c in row if c != 0)
                output_colors = set(c for row in output_grid for c in row if c != 0)
                inferred_boundary = min(output_colors - input_colors) if (output_colors - input_colors) else 2
                recipe.add_step('mark_boundary', boundary_color=inferred_boundary)
                recipe.confidence = accuracy
                return recipe
        except Exception:
            pass
        
        # Try outline with various colors
        for outline_color in range(1, 10):
            try:
                result = outline(input_grid, outline_color=outline_color)
                match, accuracy = self._grids_match(result, output_grid, tolerance)
                if match:
                    recipe = TransformationRecipe()
                    recipe.add_step('outline', outline_color=outline_color)
                    recipe.confidence = accuracy
                    return recipe
            except Exception:
                continue
        
        # Try color_swap
        for c1 in all_colors:
            for c2 in all_colors:
                if c1 >= c2:
                    continue
                try:
                    result = color_swap(input_grid, c1, c2)
                    match, accuracy = self._grids_match(result, output_grid, tolerance)
                    if match:
                        recipe = TransformationRecipe()
                        recipe.add_step('color_swap', color1=c1, color2=c2)
                        recipe.confidence = accuracy
                        return recipe
                except Exception:
                    continue
        
        # Try color_replace
        for old_c in colors_in_input:
            for new_c in all_colors:
                if old_c == new_c:
                    continue
                try:
                    result = color_replace(input_grid, old_c, new_c)
                    match, accuracy = self._grids_match(result, output_grid, tolerance)
                    if match:
                        recipe = TransformationRecipe()
                        recipe.add_step('color_replace', old_color=old_c, new_color=new_c)
                        recipe.confidence = accuracy
                        return recipe
                except Exception:
                    continue
        
        # Check identity last
        match, accuracy = self._grids_match(input_grid, output_grid, tolerance)
        if match:
            recipe = TransformationRecipe()
            recipe.add_step('identity')
            recipe.confidence = accuracy
            return recipe
        
        return None
    
    def discover_composed_transform(
        self,
        input_grid: Grid,
        output_grid: Grid,
        max_steps: int = 2,
        tolerance: float = 0.95,
        inferred_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Optional[TransformationRecipe]:
        """
        Try to find a composed transformation (2+ steps).
        
        Args:
            input_grid: Input grid
            output_grid: Target output
            max_steps: Maximum steps in composition
            tolerance: Accuracy threshold
            inferred_params: Optional dict of template_name -> params to try
        """
        if max_steps < 2:
            return None
        
        # Get list of transforms to try
        transforms_to_try = list(self.ATOMIC_TRANSFORMS.keys())
        transforms_to_try.remove('identity')
        
        # First, try parameterized combinations if we have inferred params
        if inferred_params:
            # Extract colors for morphological ops
            boundary_colors = []
            translation_amounts = []
            
            for name, params in inferred_params.items():
                if 'boundary_color' in params:
                    boundary_colors.append(params['boundary_color'])
                if 'amount' in params:
                    translation_amounts.append(params['amount'])
                if 'dy' in params or 'dx' in params:
                    dy = abs(params.get('dy', 0))
                    dx = abs(params.get('dx', 0))
                    translation_amounts.extend([dy, dx])
            
            boundary_colors = list(set(boundary_colors)) or [2]
            translation_amounts = list(set(translation_amounts)) or [1]
            
            # Try morph transforms with correct boundary colors
            morph_transforms = ['mark_boundary', 'mark_boundary_8conn', 'morph_outline_recolor']
            geometric_transforms = ['rotate_90', 'rotate_180', 'rotate_270',
                                   'flip_horizontal', 'flip_vertical']
            
            # Composition: geometric + morph with params
            for geom in geometric_transforms:
                func1 = self.ATOMIC_TRANSFORMS[geom][0]
                try:
                    intermediate = func1(input_grid)
                except Exception:
                    continue
                
                for morph in morph_transforms:
                    for bc in boundary_colors:
                        try:
                            result = self.apply_transform(intermediate, morph, boundary_color=bc)
                            match, accuracy = self._grids_match(result, output_grid, tolerance)
                            if match:
                                recipe = TransformationRecipe()
                                recipe.add_step(geom)
                                recipe.add_step(morph, boundary_color=bc)
                                recipe.confidence = accuracy
                                return recipe
                        except Exception:
                            continue
            
            # Composition: morph with params + geometric
            for morph in morph_transforms:
                for bc in boundary_colors:
                    try:
                        intermediate = self.apply_transform(input_grid, morph, boundary_color=bc)
                    except Exception:
                        continue
                    
                    for geom in geometric_transforms:
                        func2 = self.ATOMIC_TRANSFORMS[geom][0]
                        try:
                            result = func2(intermediate)
                            match, accuracy = self._grids_match(result, output_grid, tolerance)
                            if match:
                                recipe = TransformationRecipe()
                                recipe.add_step(morph, boundary_color=bc)
                                recipe.add_step(geom)
                                recipe.confidence = accuracy
                                return recipe
                        except Exception:
                            continue
            
            # Composition: translate + morph / morph + translate
            translate_transforms = ['translate_up', 'translate_down', 'translate_left', 'translate_right']
            for trans in translate_transforms:
                for amount in translation_amounts:
                    try:
                        intermediate = self.apply_transform(input_grid, trans, amount=amount)
                    except Exception:
                        continue
                    
                    for morph in morph_transforms:
                        for bc in boundary_colors:
                            try:
                                result = self.apply_transform(intermediate, morph, boundary_color=bc)
                                match, accuracy = self._grids_match(result, output_grid, tolerance)
                                if match:
                                    recipe = TransformationRecipe()
                                    recipe.add_step(trans, amount=amount)
                                    recipe.add_step(morph, boundary_color=bc)
                                    recipe.confidence = accuracy
                                    return recipe
                            except Exception:
                                continue
        
        # Fall back to default parameter combinations
        # Try all pairs with default params
        for name1 in transforms_to_try:
            func1, _ = self.ATOMIC_TRANSFORMS[name1]
            try:
                intermediate = func1(input_grid)
            except Exception:
                continue
            
            for name2 in transforms_to_try:
                func2, _ = self.ATOMIC_TRANSFORMS[name2]
                try:
                    result = func2(intermediate)
                    match, accuracy = self._grids_match(result, output_grid, tolerance)
                    if match:
                        recipe = TransformationRecipe()
                        recipe.add_step(name1)
                        recipe.add_step(name2)
                        recipe.confidence = accuracy
                        return recipe
                except Exception:
                    continue
        
        # Try 3-step if max_steps >= 3
        if max_steps >= 3:
            for name1 in transforms_to_try[:10]:  # Limit for performance
                func1, _ = self.ATOMIC_TRANSFORMS[name1]
                try:
                    int1 = func1(input_grid)
                except Exception:
                    continue
                
                for name2 in transforms_to_try[:10]:
                    func2, _ = self.ATOMIC_TRANSFORMS[name2]
                    try:
                        int2 = func2(int1)
                    except Exception:
                        continue
                    
                    for name3 in transforms_to_try[:10]:
                        func3, _ = self.ATOMIC_TRANSFORMS[name3]
                        try:
                            result = func3(int2)
                            match, accuracy = self._grids_match(result, output_grid, tolerance)
                            if match:
                                recipe = TransformationRecipe()
                                recipe.add_step(name1)
                                recipe.add_step(name2)
                                recipe.add_step(name3)
                                recipe.confidence = accuracy
                                return recipe
                        except Exception:
                            continue
        
        return None
    
    def _infer_params_from_grids(
        self,
        input_grid: Grid,
        output_grid: Grid
    ) -> Dict[str, Dict[str, Any]]:
        """
        Infer transformation parameters from input/output grid comparison.
        
        Returns:
            Dict mapping transform names to their inferred parameters
        """
        inferred = {}
        
        if not input_grid or not output_grid:
            return inferred
        
        # Infer boundary color for morphological operations
        input_colors = set()
        output_colors = set()
        for row in input_grid:
            input_colors.update(c for c in row if c != 0)
        for row in output_grid:
            output_colors.update(c for c in row if c != 0)
        
        new_colors = output_colors - input_colors
        if new_colors:
            boundary_color = min(new_colors)  # Use smallest new color
            interior_color = min(input_colors) if input_colors else 1
            inferred['mark_boundary'] = {
                'boundary_color': boundary_color,
                'interior_color': interior_color
            }
            inferred['mark_boundary_8conn'] = {
                'boundary_color': boundary_color,
                'interior_color': interior_color
            }
            inferred['morph_outline_recolor'] = {
                'boundary_color': boundary_color
            }
        
        # Infer translation offset
        translation = detect_translation(input_grid, output_grid)
        if translation:
            dy, dx = translation
            inferred['translate_object'] = {'dy': dy, 'dx': dx}
            
            # Map to directional translations
            if dy < 0 and dx == 0:
                inferred['translate_up'] = {'amount': abs(dy)}
            elif dy > 0 and dx == 0:
                inferred['translate_down'] = {'amount': dy}
            elif dy == 0 and dx < 0:
                inferred['translate_left'] = {'amount': abs(dx)}
            elif dy == 0 and dx > 0:
                inferred['translate_right'] = {'amount': dx}
            elif dy < 0 and dx < 0:
                inferred['translate_up_left'] = {'amount': max(abs(dy), abs(dx))}
            elif dy < 0 and dx > 0:
                inferred['translate_up_right'] = {'amount': max(abs(dy), abs(dx))}
            elif dy > 0 and dx < 0:
                inferred['translate_down_left'] = {'amount': max(abs(dy), abs(dx))}
            elif dy > 0 and dx > 0:
                inferred['translate_down_right'] = {'amount': max(abs(dy), abs(dx))}
        
        return inferred
    
    def discover_transformation(
        self,
        input_grid: Grid,
        output_grid: Grid,
        max_steps: int = 2,
        tolerance: float = 0.95,
        inferred_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Optional[TransformationRecipe]:
        """
        Discover a transformation recipe that converts input to output.
        
        Args:
            input_grid: Input grid
            output_grid: Target output grid
            max_steps: Maximum transformation steps
            tolerance: Accuracy threshold for match
            inferred_params: Optional pre-computed parameters (for batch processing)
        
        Returns:
            TransformationRecipe or None if not found
        """
        # Check cache
        cache_key = (self._grid_hash(input_grid), self._grid_hash(output_grid))
        if cache_key in self.recipe_cache:
            return self.recipe_cache[cache_key]
        
        # Infer parameters if not provided
        if inferred_params is None:
            inferred_params = self._infer_params_from_grids(input_grid, output_grid)
        
        # Try single transform first
        recipe = self.discover_single_transform(input_grid, output_grid, tolerance)
        if recipe:
            self.recipe_cache[cache_key] = recipe
            return recipe
        
        # Try composed transforms with inferred parameters
        recipe = self.discover_composed_transform(
            input_grid, output_grid, max_steps, tolerance, inferred_params
        )
        if recipe:
            self.recipe_cache[cache_key] = recipe
            return recipe
        
        return None
    
    def _merge_inferred_params(
        self,
        params_list: List[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge inferred parameters from multiple training pairs.
        
        Uses majority voting for each parameter to find consistent values.
        """
        if not params_list:
            return {}
        
        merged = {}
        
        # Collect all transform names
        all_transforms = set()
        for params in params_list:
            all_transforms.update(params.keys())
        
        for transform_name in all_transforms:
            # Collect all parameter sets for this transform
            all_param_sets = []
            for params in params_list:
                if transform_name in params:
                    all_param_sets.append(params[transform_name])
            
            if not all_param_sets:
                continue
            
            # Merge parameters using majority voting
            merged_params = {}
            all_param_keys = set()
            for ps in all_param_sets:
                all_param_keys.update(ps.keys())
            
            for key in all_param_keys:
                values = [ps.get(key) for ps in all_param_sets if key in ps]
                if values:
                    # Use most common value (for int/color params)
                    from collections import Counter
                    most_common = Counter(values).most_common(1)[0][0]
                    merged_params[key] = most_common
            
            merged[transform_name] = merged_params
        
        return merged
    
    def discover_from_examples(
        self,
        train_pairs: List[Dict[str, Grid]],
        max_steps: int = 2,
        tolerance: float = 0.90
    ) -> Optional[TransformationRecipe]:
        """
        Discover a consistent transformation from multiple examples.
        
        Args:
            train_pairs: List of {"input": Grid, "output": Grid}
            max_steps: Maximum transformation steps
            tolerance: Accuracy threshold
        
        Returns:
            Recipe that works for most examples, or None
        """
        if not train_pairs:
            return None
        
        # Collect inferred parameters from all training pairs
        all_inferred_params = []
        for pair in train_pairs:
            params = self._infer_params_from_grids(pair["input"], pair["output"])
            all_inferred_params.append(params)
        
        # Merge parameters using majority voting
        merged_params = self._merge_inferred_params(all_inferred_params)
        
        # Try discovery using first pair with merged parameters
        candidate = self.discover_transformation(
            train_pairs[0]["input"],
            train_pairs[0]["output"],
            max_steps,
            tolerance,
            inferred_params=merged_params
        )
        
        if candidate:
            # Verify on all pairs
            success_count = 0
            total_accuracy = 0.0
            
            for pair in train_pairs:
                result = self.apply_recipe(pair["input"], candidate)
                match, accuracy = self._grids_match(result, pair["output"], tolerance)
                if match:
                    success_count += 1
                total_accuracy += accuracy
            
            # Accept if works for majority
            if success_count >= len(train_pairs) * 0.5:
                candidate.confidence = total_accuracy / len(train_pairs)
                return candidate
        
        # Try other pairs as base with merged parameters
        for i in range(1, min(3, len(train_pairs))):
            candidate = self.discover_transformation(
                train_pairs[i]["input"],
                train_pairs[i]["output"],
                max_steps,
                tolerance,
                inferred_params=merged_params
            )
            
            if not candidate:
                continue
            
            # Verify
            success_count = 0
            total_accuracy = 0.0
            
            for pair in train_pairs:
                result = self.apply_recipe(pair["input"], candidate)
                match, accuracy = self._grids_match(result, pair["output"], tolerance - 0.1)
                if match:
                    success_count += 1
                total_accuracy += accuracy
            
            if success_count >= len(train_pairs) * 0.5:
                candidate.confidence = total_accuracy / len(train_pairs)
                return candidate
        
        # Fallback: try without merged params (original behavior)
        for i in range(min(3, len(train_pairs))):
            candidate = self.discover_transformation(
                train_pairs[i]["input"],
                train_pairs[i]["output"],
                max_steps,
                tolerance
            )
            
            if not candidate:
                continue
            
            success_count = 0
            total_accuracy = 0.0
            
            for pair in train_pairs:
                result = self.apply_recipe(pair["input"], candidate)
                match, accuracy = self._grids_match(result, pair["output"], tolerance - 0.1)
                if match:
                    success_count += 1
                total_accuracy += accuracy
            
            if success_count >= len(train_pairs) * 0.5:
                candidate.confidence = total_accuracy / len(train_pairs)
                return candidate
        
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def list_all_transforms() -> List[str]:
    """Get list of all available transformation names."""
    return list(GridTemplateEngine.ATOMIC_TRANSFORMS.keys()) + GridTemplateEngine.COLOR_TRANSFORMS


def create_recipe(*steps: str) -> TransformationRecipe:
    """Create a recipe from step names."""
    recipe = TransformationRecipe()
    for step in steps:
        recipe.add_step(step)
    return recipe


if __name__ == '__main__':
    # Demo
    print("=== Grid Template Engine Demo ===\n")
    
    # Test grid
    test_grid = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    
    print("Original grid:")
    for row in test_grid:
        print(row)
    
    print("\nAfter rotate_90:")
    rotated = rotate_90(test_grid)
    for row in rotated:
        print(row)
    
    print("\nAfter gravity_down:")
    gravity = gravity_down(test_grid)
    for row in gravity:
        print(row)
    
    print("\nAfter rotate_90 → gravity_down (composed):")
    composed = gravity_down(rotate_90(test_grid))
    for row in composed:
        print(row)
    
    # Test discovery
    print("\n=== Transformation Discovery ===")
    
    # Create a simple HDC mock for testing
    class MockHDC:
        pass
    
    class MockEncoder:
        pass
    
    engine = GridTemplateEngine(MockHDC(), MockEncoder())
    
    # Test: input rotated 90 = output
    input_g = [[1, 2], [3, 4]]
    output_g = rotate_90(input_g)
    
    print(f"Input: {input_g}")
    print(f"Output: {output_g}")
    
    recipe = engine.discover_transformation(input_g, output_g)
    if recipe:
        print(f"Discovered: {recipe}")
        print(f"Confidence: {recipe.confidence:.2%}")
    
    # Test composed
    input_g2 = [[0, 1], [1, 1], [0, 1]]
    output_g2 = gravity_down(flip_horizontal(input_g2))
    
    print(f"\nInput: {input_g2}")
    print(f"Output: {output_g2}")
    
    recipe2 = engine.discover_transformation(input_g2, output_g2, max_steps=2)
    if recipe2:
        print(f"Discovered: {recipe2}")
        print(f"Confidence: {recipe2.confidence:.2%}")
    
    print("\n✅ Demo complete!")
    print(f"\nAvailable transforms: {list_all_transforms()}")
