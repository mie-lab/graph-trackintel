"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ""
if os.path.exists("README.md"):
    with open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name="graph-trackintel",
    version="0.0.6",
    description="Tools for creating, processing and analyzing location graphs",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="MIE Lab",
    author_email=("nwiedemann@ethz.ch"),
    license="MIT",
    url="https://github.com/mie-lab/graph-trackintel",
    install_requires=["numpy", "scipy", "psycopg2", "networkx", "pyproj", "matplotlib", "trackintel"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("."),
    python_requires=">=3.6",
    scripts=scripts,
)
