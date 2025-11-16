"""Setup script for Medical Q&A System"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="medical-qa-system",
    version="1.0.0",
    description="Medical Question Answering System using Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/Chatbot-for-Medical-Chekup-Assisstance",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "simpletransformers>=0.70.0",
        "transformers>=4.31.0",
        "torch>=2.0.0",
        "pandasql>=0.7.3",
        "rouge>=1.0.1",
        "nltk>=3.8.0",
        "wandb>=0.15.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "medical-qa-train=scripts.train:main",
            "medical-qa-pipeline=scripts.run_pipeline:main",
            "medical-qa-query=scripts.query_data:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

