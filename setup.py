from setuptools import setup


setup(
    name="EOS",
    version="0.1.0",
    author="Juan D. Velasquez",
    author_email="jdvelasq@unal.edu.co",
    license="MIT",
    url="http://github.com/jdvelasq/evolutionary-optimization-suite",
    description="Evolutionary Algorithms Suite",
    long_description="Evolutionary Optimization Suite",
    keywords="Optimization",
    platforms="any",
    provides=["EOS"],
    install_requires=[
        "matplotlib",
        "numpy",
    ],
    packages=["EOS"],
    package_dir={"EOS": "EOS"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
