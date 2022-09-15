import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="fpell",
    version="0.0.1",
    # install_requires=[
    #     "requests", "omegaconf"
    # ],
    # entry_points={
    #     'console_scripts': [
    #         'corona=corona:main',
    #     ],
    # },
    author="ranchantan",
    author_email="propella@example.com",
    description="FPELL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com",
    packages=setuptools.find_packages(),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.7',
)