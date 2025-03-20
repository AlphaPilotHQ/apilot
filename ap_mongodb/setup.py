from setuptools import setup, find_packages

setup(
    name="apilot_mongodb",
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        "pymongo>=3.12.0",
    ],
    description="MongoDB database adapter for APilot trading platform",
    long_description="MongoDB database adapter for APilot quantitative trading framework",
    author="AlphaPilot",
    author_email="alpha@example.com",
    license="MIT",
    url="https://github.com/yourusername/apilot_mongodb",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: MIT License",
    ],
)