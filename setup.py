from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
        name="mechanoChemML",
        version="0.0.1",
        description="A machine learning software library for computational materials physics",
        long_description=long_description,
        long_description_content_type="text/markdown",
        py_modules=["mechanoChemML"],
        url='https://github.com/mechanoChem/mechanoChemML',
        author='Xiaoxuan Zhang',
        author_email='zhangxiaoxuan258@gmail.com',
        package_dir={"":"mechanoChemML"},
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
            ],
        install_requires=[
          'numpy',
          ],
        extras_require = {
            "dev":[
                "pytest>=3.6",
                ],
            },
        )
