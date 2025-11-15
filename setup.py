from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements.txt and return a clean list of packages.
    Removes newline characters and ignores empty lines.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]

    return requirements


setup(
    name="ML_PROJECT_DEMO",              # change this
    version="0.1.0",
    author="Ayush",
    author_email="ayush25s9gautam@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
