"""
Autonomous Drug Discovery Lab
Self-driving laboratory framework for pharmaceutical research
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="autonomous-drug-discovery-lab",
    version="1.0.0",
    author="Oluwaseun O. Ajayi",
    author_email="seunolanikeajayi@gmail.com",
    description="Self-driving laboratory framework for autonomous pharmaceutical research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Oluwaseun-O-Ajayi/autonomous-drug-discovery-lab",
    project_urls={
        "Bug Tracker": "https://github.com/Oluwaseun-O-Ajayi/autonomous-drug-discovery-lab/issues",
        "Documentation": "https://github.com/Oluwaseun-O-Ajayi/autonomous-drug-discovery-lab/docs",
        "Source Code": "https://github.com/Oluwaseun-O-Ajayi/autonomous-drug-discovery-lab",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.7.0",
        ],
    },
    include_package_data=True,
    keywords=[
        "self-driving-labs",
        "laboratory-automation",
        "drug-discovery",
        "bayesian-optimization",
        "autonomous-experimentation",
        "pharmaceutical-research",
    ],
)