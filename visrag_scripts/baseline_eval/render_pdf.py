"""Content-addressed, page-resumable PDF rendering."""
from pathlib import Path
import fitz
from PIL import Image
from io_utils import atomic_path, file_digest, fingerprint


def get_pdf_page_count(pdf_path):
    """Returns the physical page count."""
    with fitz.open(pdf_path) as document:
        return len(document)


def render_pdf_to_images(pdf_path, output_dir, dpi=200, force_rebuild=False):
    """Reuses valid pages only for matching PDF contents and DPI."""
    identity = fingerprint([file_digest(pdf_path), dpi, "rgb-render-v2"])
    target = Path(output_dir) / identity
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    with fitz.open(pdf_path) as document:
        if not len(document):
            raise ValueError(f"Empty PDF: {pdf_path}")
        for index in range(len(document)):
            path = target / f"page_{index + 1}.png"
            valid = False
            if path.is_file() and not force_rebuild:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    valid = True
                except (OSError, SyntaxError):
                    pass
            if not valid:
                pixmap = document[index].get_pixmap(
                    dpi=dpi, colorspace=fitz.csRGB, alpha=False)
                with atomic_path(path) as temporary:
                    pixmap.save(str(temporary))
            paths.append(str(path.resolve()))
    return paths
