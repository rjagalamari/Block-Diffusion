from setuptools import setup, find_packages

setup(
    name="block_diffusion",
    version="0.1.0", 
    description="BlockDiffusion - External Language Model for XLM framework",
    packages=find_packages(),
    install_requires=[
        "xlm-core",
        "einops",
    ],
    package_data={
        "block_diffusion": ["configs/**/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.11",
    author="Your Name",
    author_email="your.email@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers", 
        "Programming Language :: Python :: 3.11",
    ],
)
