from setuptools import setup, find_packages

setup(
    name='hybridoed',  # Replace with your package's name
    version='0.1.0',  # Semantic versioning
    description='Exploring the possibility of combining OED and differentiable physics to design better inverse problems.',  # Short description
    long_description=open('README.md').read(),  # Long description from README
    long_description_content_type='text/markdown',  # If your README is in Markdown
    author='Arnaud Mercier',  # Your name
    author_email='arnaud.mercier05@gmail.com',  # Your email
    url='https://github.com/armercier/Hybrid-OED',  # URL to your project
    license='MIT',  # License type
    packages=find_packages(where='src'),  # Automatically find subpackages in src/
    package_dir={'': 'src'},  # Specify that packages are in the src/ directory
    include_package_data=True,  # Include files specified in MANIFEST.in
    python_requires='>=3.8',  # Specify the Python version
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
        # Add other dependencies here
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'sphinx>=4.0.0',
            'black',
            # Add additional development dependencies here
        ],
    },
    entry_points={
        'console_scripts': [
            'run-simulation=project_name.scripts.run_simulation:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
