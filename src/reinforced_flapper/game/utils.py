"""Utility functions for game geometry and collision."""

from collections.abc import Callable
from functools import wraps
from typing import Any

import pygame

HitMaskType = list[list[bool]]


def clamp(n: float, minn: float, maxn: float) -> float:
    """Clamp a number between two values.

    Args:
        n: The value to clamp.
        minn: Lower bound.
        maxn: Upper bound.

    Returns:
        The clamped value.
    """
    return max(min(maxn, n), minn)


def memoize(func: Callable) -> Callable:
    """Memoize a function on its arguments.

    Args:
        func: The function to memoize.

    Returns:
        The wrapped function.
    """
    cache: dict[Any, Any] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


@memoize
def get_hit_mask(image: pygame.Surface) -> HitMaskType:
    """Build a hit mask from an image's per-pixel alpha.

    Args:
        image: Surface with per-pixel alpha.

    Returns:
        Column-major grid of booleans, True where the pixel is opaque.
    """
    return [
        [bool(image.get_at((x, y))[3]) for y in range(image.get_height())]
        for x in range(image.get_width())
    ]


def pixel_collision(
    rect1: pygame.Rect,
    rect2: pygame.Rect,
    hitmask1: HitMaskType,
    hitmask2: HitMaskType,
) -> bool:
    """Check whether two objects overlap on opaque pixels, not just rects.

    Args:
        rect1: The first object's rect.
        rect2: The second object's rect.
        hitmask1: The first object's hit mask.
        hitmask2: The second object's hit mask.

    Returns:
        True if any opaque pixels overlap.
    """
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
