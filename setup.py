from setuptools import setup, find_packages
import subprocess
import re

result = subprocess.run(['git', 'branch', '--show-current'], stdout=subprocess.PIPE)
branch = result.stdout.decode('utf-8').strip()

if branch == 'develop':
    version = 'develop'
else:
    version = re.search(r'release-(\d+\.\d+)', branch).group(1)

# requirements.txt 파일을 읽고 install_requires를 설정
with open('requirements.txt', 'r') as file:
    requirements = file.readlines()

requirements = [line.strip() for line in requirements if line.strip()]

install_requires = []
for req in requirements:
    package_name, *constraints = req.split()
    if constraints:
        install_requires.append(f"{package_name}{constraints[0]}")
    else:
        install_requires.append(package_name)

setup(
    name='alolib',
    version=version,
    author='wonjun.sung',
    author_email='wonjun.sung@lge.com',
    packages=find_packages('.'),
    install_requires=install_requires,
    license='MIT',
    description='AI Platform Common Libraries'

)