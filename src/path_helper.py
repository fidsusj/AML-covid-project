""" Helper function for getting path to project root """

from pathlib import Path


def get_project_root() -> Path:
    """ return path to the project root"""
    return Path(__file__).parent