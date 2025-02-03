from setuptools import find_packages, setup

# Package metadata
NAME = "DLOSeg"
VERSION = "0.1"
DESCRIPTION = "DLOSeg: Deep Learning Object Segmentation"
URL = "https://github.com/rotematari/DLOSeg"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your.email@example.com"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=1.7.1",
    "torchvision>=0.8.2",
    "numpy>=1.19.2",
    "tqdm>=4.50.2",
    # "clipseg @ git+https://github.com/timojl/clipseg.git",
    # "sam2RT @ git+https://github.com/Gy920/segment-anything-2-real-time.git",
    # Add other dependencies here
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.6",
)