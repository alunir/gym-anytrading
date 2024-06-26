from setuptools import setup, find_packages

setup(
    name="gym_anytrading",
    version="2.2.4",
    packages=find_packages(),
    author="AminHP",
    author_email="mdan.hagh@gmail.com",
    license="MIT",
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.16.4",
        "pandas>=0.24.2",
        "pyts>=0.13.0",
        "matplotlib>=3.1.1",
        "pyts>=0.13.0",
    ],
    package_data={"gym_anytrading": ["datasets/data/*"]},
)
