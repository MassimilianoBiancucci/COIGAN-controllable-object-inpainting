#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = [
    "kaggle==1.5.12",
    "numpy==1.23.3",
    "opencv-python-headless==4.6.0.66",
    "wandb==0.13.7",
    "pbjson==1.15",
    "hydra-core==1.2",
    "omegaconf==2.2.3",
    "kornia==0.6.8",
    "pandas==1.5.0",
    "scipy==1.9.1",
    "matplotlib==3.6.0",
]

test_requirements = []

dependency_links = [
]

setup(
    author="Massimiliano Biancucci",
    author_email="Binacucci95@Gmail.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    description="Training, inference and test pipeline package for a controled masked adversarial generator",
    install_requires=requirements,
    long_description="",
    include_package_data=True,
    keywords="COIGAN",
    name="COIGAN",
    packages=find_packages(
        include=[
            "evaluation",
            "evaluation.*",
            "modules",
            "modules.*",
            "training",
            "training.*",
            "inference",
            "inference.*",
            "shape_training",
            "shape_training.*",
            "utils",
            "utils.*",
        ]
    ),
    dependency_links=dependency_links,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/MassimilianoBiancucci/COIGAN-controllable-object-inpainting",
    version="0.1.0",
    zip_safe=False,
)
