from setuptools import setup, find_packages

setup(
    name="pyofn",
    version="0.1.0",
    description="Ordered Fuzzy Numbers (Skierowane Liczby Rozmyte) — biblioteka Python",
    author="pyofn contributors",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5"],
        "dev": ["pytest", "matplotlib>=3.5"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
