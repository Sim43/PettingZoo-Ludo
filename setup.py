from setuptools import find_packages, setup

setup(
    name="ludo-pettingzoo",
    version='0.1.0',
    description="Ludo board game environment for PettingZoo",
    author="Muhammad Asim Khan",
    email="masimwork43@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pettingzoo>=1.24.0",
        "gymnasium>=1.0.0",
        "numpy>=1.21.0",
        "pygame>=2.1.0",
    ]
)
