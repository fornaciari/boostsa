#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()
#
with open("README.rst", "r") as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="boostsa",
    version="0.1.7",
    author="Tommaso Fornaciari",
    author_email="fornaciari@unibocconi.it",
    description="A package to compute bootstrap sampling significance test",
    long_description=readme, # readme + '\n\n' + history,
    long_description_content_type="text/x-rst",
    url="https://github.com/fornaciari/bootstrap",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.6',
    license="MIT license",
)


    # long_description_content_type="text/markdown",


# setup_requirements = ['pytest-runner', ]
#
# test_requirements = ['pytest>=3', ]
#
# setup(
#     classifiers=[
#         'Development Status :: 2 - Pre-Alpha',
#         'Intended Audience :: Developers',
#         'License :: OSI Approved :: MIT License',
#         'Natural Language :: English',
#         'Programming Language :: Python :: 3',
#         'Programming Language :: Python :: 3.5',
#         'Programming Language :: Python :: 3.6',
#         'Programming Language :: Python :: 3.7',
#         'Programming Language :: Python :: 3.8',
#     ],
#     include_package_data=True,
#     keywords='quica',
#     packages=find_packages(include=['quica', 'quica.*']),
#     setup_requires=setup_requirements,
#     test_suite='tests',
#     tests_require=test_requirements,
#     zip_safe=False,
# )


