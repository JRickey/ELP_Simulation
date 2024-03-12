from setuptools import setup
from pathlib import Path
import glob

APP = ['simulator.py']
DATA_FILES = []
base_path = Path.cwd()
directories = [
    'cloglog',
    'loans',
    'msa_data',
    'rate_HPI_process'
]
for directory in directories:
    dir_path = base_path / directory  # Use / operator for path joining
    # Use glob with pathlib Path objects
    # Note: rglob(pattern) recursively searches for files matching the pattern
    files = [str(file) for file in dir_path.rglob('*.csv')]
    if files:
        # DATA_FILES expects a tuple with the first element being the destination directory name
        # and the second being a list of file paths
        DATA_FILES.append((directory, files))

OPTIONS = {
    'argv_emulation': True,
    'plist': {
        'CFBundleName': 'Parallel Simulator',
        'CFBundleShortVersionString': '0.1.0',
        'CFBundleVersion': '0.1.0',
    },
    'packages': ['pandas'],  # Assuming you're using pandas, include it here
    'resources': DATA_FILES,  # This line might not be necessary, depending on py2app version
}

setup(
    app=APP,
    name='Parallel Simulator',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
