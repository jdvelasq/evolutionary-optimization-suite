from setuptools import setup


setup(
    name="EAOS",
    version="0.0.1",
    author="Juan D. Velasquez",
    author_email="jdvelasq@unal.edu.co",
    license="MIT",
    url="http://github.com/jdvelasq/evolutionary-optimization-suite",
    description="Evolutionary Algorithms Suite",
    long_description="Evolutionary Algorithms Optimization Suite",
    keywords="Optimization",
    platforms="any",
    provides=["EAOS"],
    install_requires=[
        "matplotlib",
        "numpy",
    ],
    packages=["EAOS"],
    package_dir={"EAOS": "EAOS"},
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
