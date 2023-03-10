import io
import os
import re

import setuptools

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

print(BASE_DIR)

def get_long_description():
    with io.open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    try:
        with open(os.path.join(BASE_DIR, "requirements.txt"), encoding="utf-8") as f:
            return f.read().split("\n")
    except:
        return []


def get_version():
    version_file = os.path.join(BASE_DIR, "backbones_unet", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setuptools.setup(
    name="pretrained-backbones-unet",
    version=get_version(),
    author="Berkay Mayali",
    license="MIT",
    description="Packaged version of the timm pretrained model backbones for Unet architecture",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/mberkay0/pretrained-backbones-unet",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning, deep-learning, pytorch, vision, semantic-segmentation, image-pixel-classification, object-detection, convnext, unet",
)