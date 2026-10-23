"""PDF rendering utility to rasterize PDF pages into PNG images.

Supports caching to prevent redundant rendering and PyMuPDF (fitz) with 200 DPI.
"""

import os
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from PIL import Image


def get_pdf_page_count(pdf_path: str) -> int:
    """Returns the total number of pages in the given PDF file."""
    with fitz.open(pdf_path) as doc:
        return len(doc)


def render_pdf_to_images(
    pdf_path: str,
    output_dir: Path,
    dpi: int = 200,
    force_rebuild: bool = False,
) -> List[str]:
    """Renders all pages of a PDF to images with caching.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Base directory where page images will be cached.
        dpi: Resolution in DPI (default 200).
        force_rebuild: If True, re-renders even if images exist.

    Returns:
        List of absolute file paths to the rendered page images, 1-indexed in order.
    """
    pdf_name = Path(pdf_path).stem
    target_dir = output_dir / pdf_name
    target_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        image_paths: List[str] = []

        # Check if already fully rendered
        existing_images = [
            target_dir / f"page_{page_num:04d}.png"
            for page_num in range(1, total_pages + 1)
        ]
        if not force_rebuild and all(p.exists() for p in existing_images):
            return [str(p.resolve()) for p in existing_images]

        # Render missing pages
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            out_img_path = target_dir / f"page_{page_num:04d}.png"
            if force_rebuild or not out_img_path.exists():
                pix = page.get_pixmap(dpi=dpi)
                pix.save(str(out_img_path))
            image_paths.append(str(out_img_path.resolve()))

    return image_paths
