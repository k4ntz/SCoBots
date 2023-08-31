from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()


setup(
    name='scobi',
    version='0.0.0',
    author='',
    author_email='',
    packages=find_packages(),
    include_package_data=True,
    description='Successive Concepet Bottleneck Interface',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements
)
