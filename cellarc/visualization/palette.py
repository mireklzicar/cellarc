"""Shared colour palettes for Cell ARC visualisations."""

from __future__ import annotations

from matplotlib.colors import ListedColormap

CMAP_HEX = [
    "#252525",
    "#0074D9",
    "#FF4136",
    "#37D449",
    "#FFDC00",
    "#E6E6E6",
    "#F012BE",
    "#FF871E",
    "#54D2EB",
    "#8D1D2C",
    "#FFFFFF",
]

BG_COLOR = "#EEEFF6"

PALETTE = ListedColormap(CMAP_HEX)
PALETTE.set_bad(color=BG_COLOR)


__all__ = ["CMAP_HEX", "BG_COLOR", "PALETTE"]
