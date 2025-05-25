from setuptools import find_packages, setup

setup(
    name="image_similarity",
    version="0.1.0",
    description="A project for image similarity tasks",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow",
        "pandas",
        "numpy",
        "matplotlib",
        "opencv-python",
        "scikit-learn",
        "Pillow",
        "seaborn",
        "basemap",
        "rignak @ git+https://github.com/AurelienColin/miscellaneous.git#egg=rignak",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
