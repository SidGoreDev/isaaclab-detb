from setuptools import find_packages, setup


setup(
    name="detb-lab",
    version="0.1.3",
    description="DETB external Isaac Lab extension package",
    package_dir={"": "."},
    packages=find_packages(where="."),
    include_package_data=True,
)
