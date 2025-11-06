# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app_cocoa.py'],
    pathex=[],
    binaries=[],
    datas=[('letter_recognition_model.pth', '.'), ('dictionary.txt', '.')],
    hiddenimports=['Quartz', 'Cocoa', 'PyObjCTools', 'objc'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WordHuntSolver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='WordHuntSolver',
)
app = BUNDLE(
    coll,
    name='WordHuntSolver.app',
    icon=None,
    bundle_identifier='com.local.wordhuntsolver',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'NSAppleEventsUsageDescription': 'WordHuntSolver needs to control iPhone Mirroring to position and focus the window.',
        'LSMinimumSystemVersion': '10.13'
    }
)
