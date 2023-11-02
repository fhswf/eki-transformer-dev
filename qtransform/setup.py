from setuptools import find_packages, setup

with open("README.md", encoding="utf8") as f:
    readme = f.read()

setup_args = dict(
    name="qtransform",
    packages=find_packages(),
    version='0.0.2dev',
    description="Various transformers with quantization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Max Kuhmichel, FH SWF",
    author_email="kuhmichel.max@fh-swf.de",
    url="https://github.com/eki-project/transformers",
    license="MIT",
    python_requires=">=3.10",
    include_package_data=True,
    entry_points={
     'console_scripts': [
        'qtransform = qtransform.__main__:cli_wrapper',
        ]
    },
)

setup_args['install_requires'] = install_requires = []
with open('requirements.txt') as f:
    for line in f.readlines():
        req = line.strip()
        if not req or req.startswith(('-e', '#')):
            continue
        install_requires.append(req)

if __name__ == '__main__':
    setup(**setup_args)