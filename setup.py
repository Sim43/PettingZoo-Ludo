from setuptools import find_packages, setup

setup(
    name="ludo-pettingzoo",
    version='0.1.0',
    description="Ludo board game environment for PettingZoo",
    author="Muhammad Asim Khan",
    author_email="masimwork43@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pettingzoo>=1.24.0",
        "gymnasium>=1.0.0",
        "numpy>=1.21.0",
        "pygame>=2.1.0",
        "pytest>=7.0.0",
        "jinja2>=3.0.0",
        "typeguard>=3.0.0",
        "setuptools>=69.0.0",
        "lark",
    ]
)
