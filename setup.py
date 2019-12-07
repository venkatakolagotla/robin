#!/usr/bin/python3

import setuptools

with open("README.md", "r") as f:
    robin_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="robin",
    version="0.1.0",
    author="Venkata Kolagotla",
    author_email="venkata.kolagotla@gmail.com",
    description="A package to perform U-net binarization on doc images",
    long_description=robin_description,
    url="https://github.com/venkatakolagotla/robin",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required
)
