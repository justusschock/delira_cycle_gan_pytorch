import os
import re
from setuptools import find_packages, setup


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


def find_version(file):
    content = read_file(file)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content,
                              re.M)
    if version_match:
        return version_match.group(1)


readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))
license = read_file(os.path.join(os.path.dirname(__file__), "LICENSE"))
cycle_gan_version = find_version(os.path.join(os.path.dirname(__file__),
                                              "cycle_gan",
                                              "__init__.py"))

setup(
    name='delira_cycle_gan',
    version=cycle_gan_version,
    packages=find_packages(),
    url='https://github.com/justusschock/delira_cycle_gan_pytorch',
    license=license,
    author='Justus Schock',
    author_email='justus.schock@rwth-aachen.de',
    description=''
)
