from setuptools import setup, find_packages

setup(
    name="tgpr-ucb",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.11.0",
        "accelerate>=0.30.0",
        "datasets>=2.19.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0.1",
        "pandas>=2.0.0",
        "pyarrow>=15.0.0",
        "tqdm",
    ],
    python_requires=">=3.9",
)
