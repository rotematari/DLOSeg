from setuptools import setup, find_packages

setup(
    name="dloseg",
    version="0.1.0",
    description="",
    author="rotematari",
    author_email="rotem.atri@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Install clipseg from GitHub (using the 'main' branch; change if needed)
        # "clipseg @ git+https://github.com/timojl/clipseg.git@main",
        # "sam2RT @ git+https://github.com/Gy920/segment-anything-2-real-time.git@main",

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)