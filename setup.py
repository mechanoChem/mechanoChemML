from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    required = f.read()

setup(
        name="mechanoChemML",
        version="0.0.5",
        packages=find_packages(),
        description="A machine learning software library for computational materials physics",
        long_description=long_description,
        long_description_content_type="text/markdown",
        py_modules=["mechanoChemML"],
        url='https://github.com/mechanoChem/mechanoChemML',
        author='Xiaoxuan Zhang',
        author_email='zhangxiaoxuan258@gmail.com',
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
            ],
        install_requires=required,
        extras_require = {
            "dev":[
                "pytest>=3.6",
                ],
            },
        )
