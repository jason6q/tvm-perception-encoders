from setuptools import setup, find_packages

with open("requirements.txt") as fi:
    requirements = fi.read().splitlines()
    print(requirements)

setup(
    name="TVM Perception Encoders",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements
)
