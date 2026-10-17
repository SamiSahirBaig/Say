"""Reporting, map generation, PDF brief, and PowerPoint presentation generator."""
from .map_generator import generate_district_map
from .pdf_brief import generate_inspectors_brief_pdf
from .presentation import generate_presentation_and_script

__all__ = [
    "generate_district_map",
    "generate_inspectors_brief_pdf",
    "generate_presentation_and_script",
]
