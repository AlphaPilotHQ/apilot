from setuptools import setup, find_packages

setup(
    name="vnpy_mongodb",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pymongo>=3.12.0",
    ],
    description="MongoDB database for VeighNa quant trading framework",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/vnpy_mongodb",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: MIT License",
    ],
)