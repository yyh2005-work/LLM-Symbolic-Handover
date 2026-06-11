"""Packaging configuration for the LLM-Symbolic-Handover project."""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="llm-symbolic-handover",
    version="0.1.0",
    description="LLM-assisted symbolic rule extraction and deployment for handover optimization.",
    license="MIT",
    packages=find_packages(where="src", include=["ho_optim_drl", "ho_optim_drl.*"]),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    include_package_data=True,
    zip_safe=False,
)
