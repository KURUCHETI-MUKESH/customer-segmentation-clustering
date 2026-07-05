"""
utils.py
--------
Shared helper functions used across the preprocessing, training, and
evaluation modules: path handling, plot saving, and logging helpers.
"""

import os
import logging

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Path configuration
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

RAW_DATA_PATH = os.path.join(DATA_DIR, "mall_customers.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "mall_customers_clean.csv")
CLUSTERED_DATA_PATH = os.path.join(DATA_DIR, "mall_customers_clustered.csv")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a consistent format.

    Parameters
    ----------
    name : str
        Name of the logger, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_fig(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    """Save a matplotlib figure into the project's ``images`` directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        File name (e.g. ``"elbow_method.png"``).
    dpi : int, optional
        Resolution of the saved image, by default 150.

    Returns
    -------
    str
        Full path where the image was saved.
    """
    ensure_dir(IMAGES_DIR)
    out_path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
