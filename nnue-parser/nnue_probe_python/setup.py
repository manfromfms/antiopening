from setuptools import setup, find_packages

setup(
    name='nnue_probe',
    version='0.1.0',
    description='NNUE Evaluator Library for Chess Positions',
    author='Jimmy Luong',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'python-chess',
    ],
    python_requires='>=3.6',
)
