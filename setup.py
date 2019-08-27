"""Setup script for package."""
import re
from setuptools import setup, find_packages

VERSION = re.search(
    r'^VERSION\s*=\s*"(.*)"', open("rectools/version.py").read(), re.M
).group(1)
with open("README.md", "rb") as f:
    LONG_DESCRIPTION = f.read().decode("utf-8")

setup(
    name="rectools",
    version=VERSION,
    description="Package for measuring performance of recommender system prototypes.",
    long_description=LONG_DESCRIPTION,
    author="Soren Lind Kristiansen",
    author_email="soren@gutsandglory.dk",
    url="https://github.com/sorenlind/rectools/",
    packages=find_packages(),
    install_requires=["implicit", "numpy", "scipy", "rankmetrics", "pandas", "tqdm"],
    extras_require={
        "dev": ["black", "flake8", "mypy", "pycodestyle", "pydocstyle", "rope"]
    },
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
)
