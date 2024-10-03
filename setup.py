from setuptools import setup, find_packages

setup(
    name='gapfinder',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scipy',
        'numpy',
        'matplotlib',
        'seaborn',
        'opencv-python',
        'pillow',
        'pytesseract',
        'jupyter',
    ],
)