from setuptools import setup

setup(name='sdb_grid_processor',
    version='0.1.0',
    description='Tool for processing a calculated grid of MESA sdB stars.',
    url='https://github.com/cespenar/sdb_grid_processor',
    author='Jakub Ostrowski',
    author_email='cespenar1@gmail.com',
    license='MIT',
    packages=['sdb_grid_processor'],
    install_requires=['mesa_reader', 'sqlalchemy'])
