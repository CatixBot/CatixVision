from setuptools import setup, find_packages
 
setup(name='SimpleObjectDetector',
    version='1.0.0',
    url='https://github.com/CatixBot/CatixVision',
    license='MIT',
    author='Mikhail Uskin',
    author_email='wurty@mail.ru',
    description='Simple object detector based on HSV color range masking',
    packages=find_packages(),
    install_requires=['opencv-python', 'numpy'],
    zip_safe=False)
