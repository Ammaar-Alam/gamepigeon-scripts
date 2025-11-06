from setuptools import setup

APP = ['app_cocoa.py']
DATA_FILES = ['letter_recognition_model.pth', 'dictionary.txt']
OPTIONS = {
    'argv_emulation': True,
    'plist': {
        'CFBundleName': 'WordHuntSolver',
        'CFBundleDisplayName': 'WordHuntSolver',
        'CFBundleIdentifier': 'com.local.wordhuntsolver',
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': True,
        'NSAppleEventsUsageDescription': 'WordHuntSolver needs to control iPhone Mirroring to position and focus the window.',
    },
    # Be explicit about large packages so they are bundled correctly
    'packages': [
        'numpy', 'PIL', 'mss', 'torch', 'torchvision'
    ],
    'includes': [
        'Cocoa', 'PyObjCTools', 'Quartz', 'cv2', 'pyautogui'
    ],
    'excludes': [
        'keyboard'  # avoid HIToolbox/TIS side-effects in bundled app
    ],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
