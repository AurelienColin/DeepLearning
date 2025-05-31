from setuptools import find_packages, setup

setup(
    name="deep_learning",
    version="0.1.0",
    description="A project for various deep learning trainings",
    author="Rignak",
    author_email="...",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow==2.19",
        "keras==3.10.0",
        "pandas",
        "numpy==1.26.4",
        "matplotlib",
        "opencv-python",
        "scikit-learn",
        "Pillow",
        # The rignak dependency is external (from GitHub) and needs to be installed separately.
        # e.g., pip install git+https://github.com/Rignak/rignak.git
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.12",
    ],
)
