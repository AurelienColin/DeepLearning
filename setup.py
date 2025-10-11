from setuptools import find_packages, setup

setup(
    name="ML",
    version="0.1.0",
    description="A project for various deep learning trainings",
    author="Rignak",
    author_email="...",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow==2.17",
        "pandas",
        "numpy==1.26.4",
        "matplotlib",
        "opencv-python",
        "scikit-learn",
        'basemap',
        "Pillow",
        "pytest",
        "rignak @ git+https://github.com/AurelienColin/miscellaneous.git#egg=rignak"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.12",
    ],
)
