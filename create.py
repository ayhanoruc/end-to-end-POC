import os

def create_project_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print(f"The directory {directory} already exists.")

def create_file(file_path, content=''):
    with open(file_path, 'w') as file:
        file.write(content)

def setup_project_structure(base_path, structure):
    """creates files and folders recursively"""
    for path, content in structure.items():
        full_path = os.path.join(base_path, path)
        if content is None:
            # It's a directory
            create_project_dir(full_path)
        elif isinstance(content, str):
            # It's a file with content (or empty if content is '')
            create_file(full_path, content)
        elif isinstance(content, dict):
            # It's a subdirectory with more files/folders inside
            create_project_dir(full_path)
            setup_project_structure(full_path, content)
  
base_directory = os.path.join(os.getcwd())

# Desired Project Structe
project_structure = {
    "artifacts": None,
    "src": {
        "__init__.py": "# src package",

        "constants": {
            "__init__.py": "",
            "config.yaml": "# Configuration settings",
            "cloud.yaml": "# cloud config - do not include credentials here, use environment variables w/ dotenv instead",
            "pipeline.yaml": "# pipeline constants",},

        "prediction_pipeline.py": "# prediction pipeline",
        "training_pipeline.py": "# training pipeline",
    },
    "tests": {
        "__init__.py": "# tests package",
    },
    "data": None,
    "notebooks": None,
    ".gitignore": "",
    ".dockerignore": "",
    "Dockerfile": "",
    "requirements.txt": "",
    "setup.py": """
from setuptools import setup, find_packages

setup(
    name='your_package_name',
    version='0.1',
    packages=find_packages(),
)
""",

"app.py": "# ENTRY POINT",
"README.md": "# Project Title\n\nProject description here.",
}

# Set up the project
setup_project_structure(base_directory, project_structure)

print(f"Project structure  created at {base_directory}")