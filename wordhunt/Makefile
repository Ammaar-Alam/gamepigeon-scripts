APP_NAME=WordHuntSolver
PYTHON?=python3
# Detect Homebrew Tcl/Tk prefix and versioned folders (supports 9.0 and 8.6)
TCLTK_PREFIX?=$(shell brew --prefix tcl-tk 2>/dev/null || echo /opt/homebrew/opt/tcl-tk)
TCL_LIB?=$(shell ls -d $(TCLTK_PREFIX)/lib/tcl9.0 $(TCLTK_PREFIX)/lib/tcl8.6 2>/dev/null | head -n1)
TK_LIB?=$(shell ls -d $(TCLTK_PREFIX)/lib/tk9.0  $(TCLTK_PREFIX)/lib/tk8.6  2>/dev/null | head -n1)
TCL_DYLIB?=$(shell ls $(TCLTK_PREFIX)/lib/libtcl*.dylib 2>/dev/null | head -n1)
TK_DYLIB?=$(shell ls $(TCLTK_PREFIX)/lib/libtk*.dylib  2>/dev/null | head -n1)

.PHONY: build-app clean run-gui

# Compose PyInstaller flags dynamically to avoid empty --add-binary
PYI_FLAGS_BASE=--windowed --name "$(APP_NAME)" --hidden-import tkinter \
	--add-data "$(TCL_LIB):lib/$(notdir $(TCL_LIB))" \
	--add-data "$(TK_LIB):lib/$(notdir $(TK_LIB))" \
	--add-data "letter_recognition_model.pth:." \
	--add-data "dictionary.txt:."

PYI_BIN_TCL=$(if $(strip $(TCL_DYLIB)),--add-binary "$(TCL_DYLIB):.",)
PYI_BIN_TK=$(if $(strip $(TK_DYLIB)),--add-binary "$(TK_DYLIB):.",)

build-app:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt pyinstaller
	@echo "Using Tcl/Tk from: $(TCLTK_PREFIX)"
	@echo "TCL_LIBRARY=$(TCL_LIB)"
	@echo "TK_LIBRARY=$(TK_LIB)"
	TCL_LIBRARY=$(TCL_LIB) TK_LIBRARY=$(TK_LIB) \
	$(PYTHON) -m PyInstaller $(PYI_FLAGS_BASE) $(PYI_BIN_TCL) $(PYI_BIN_TK) gui.py

run-gui:
	python gui.py

clean:
	rm -rf build dist __pycache__

# Build using python.org Python 3.11 (recommended for PyInstaller + Tk)
PY311=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
.PHONY: build-app-py311
build-app-py311:
	$(PY311) -m pip install --upgrade pip
	$(PY311) -m pip install -r requirements.txt pyinstaller
	@echo "Using Tcl/Tk from: $(TCLTK_PREFIX)"
	@echo "TCL_LIBRARY=$(TCL_LIB)"
	@echo "TK_LIBRARY=$(TK_LIB)"
	TCL_LIBRARY=$(TCL_LIB) TK_LIBRARY=$(TK_LIB) \
	$(PY311) -m PyInstaller $(PYI_FLAGS_BASE) $(PYI_BIN_TCL) $(PYI_BIN_TK) gui.py

.PHONY: open-app
open-app:
	@echo "Removing quarantine flag and opening app..."
	xattr -dr com.apple.quarantine dist/WordHuntSolver.app || true
	open dist/WordHuntSolver.app

.PHONY: build-app-cocoa
build-app-cocoa:
	$(PYTHON) -m pip install --upgrade pip || true
	$(PYTHON) -m pip install py2app pyobjc || true
	$(PYTHON) setup.py py2app -A || true
	@echo "Built Cocoa app at dist/WordHuntSolver.app"

.PHONY: build-app-cocoa-pyi
build-app-cocoa-pyi:
	$(PYTHON) -m pip install --upgrade pip || true
	# Ensure all runtime deps are present in this venv for bundling
	$(PYTHON) -m pip install -r requirements.txt pyinstaller pyobjc || true
	@if [ -f WordHuntSolver.spec ]; then \
		echo "Using existing spec file"; \
		$(PYTHON) -m PyInstaller --noconfirm WordHuntSolver.spec; \
	else \
		echo "Spec file missing; building from app_cocoa.py with inline options"; \
		$(PYTHON) -m PyInstaller --noconfirm --windowed --name "$(APP_NAME)" \
			--add-data "letter_recognition_model.pth:." \
			--add-data "dictionary.txt:." \
			--hidden-import Quartz --hidden-import Cocoa --hidden-import PyObjCTools --hidden-import objc \
			app_cocoa.py; \
	fi
	@echo "Built PyInstaller Cocoa app at dist/WordHuntSolver.app"

.PHONY: package-zip
package-zip:
	@echo "Creating distributable ZIP with resource forks preserved..."
	cd dist && ditto -c -k --sequesterRsrc --keepParent WordHuntSolver.app WordHuntSolver-macOS.zip
	@echo "ZIP at dist/WordHuntSolver-macOS.zip"

.PHONY: sign-app
sign-app:
	@echo "Clearing extended attributes and ad-hoc signing (override with CODESIGN_IDENTITY)"
	xattr -cr dist/WordHuntSolver.app || true
	/usr/bin/codesign --force --deep --options runtime --sign "${CODESIGN_IDENTITY:--}" dist/WordHuntSolver.app || true

.PHONY: verify-app
verify-app:
	@echo "codesign verification:" && /usr/bin/codesign -vvv --deep --strict dist/WordHuntSolver.app || true
	@echo "spctl assessment:" && /usr/sbin/spctl -a -vvv --type execute dist/WordHuntSolver.app || true

.PHONY: package-dmg
package-dmg:
	@echo "Creating DMG (unsigned)."
	rm -f dist/WordHuntSolver.dmg || true
	hdiutil create -volname "WordHuntSolver" -srcfolder dist/WordHuntSolver.app -ov -format UDZO dist/WordHuntSolver.dmg
	@echo "DMG at dist/WordHuntSolver.dmg"

.PHONY: build-app-cocoa-py311
build-app-cocoa-py311:
	$(PY311) -m pip install --upgrade pip || true
	$(PY311) -m pip install py2app pyobjc || true
	cd . && $(PY311) setup.py py2app
	@echo "Built Cocoa app (py2app) at dist/WordHuntSolver.app using Python 3.11"

.PHONY: run-app-console
run-app-console:
	@echo "Running app bundle binary for console logs..."
	@set -x; \
	if [ -x "dist/WordHuntSolver.app/Contents/MacOS/app_cocoa" ]; then \
	  dist/WordHuntSolver.app/Contents/MacOS/app_cocoa; \
	elif [ -x "dist/WordHuntSolver.app/Contents/MacOS/WordHuntSolver" ]; then \
	  dist/WordHuntSolver.app/Contents/MacOS/WordHuntSolver; \
	else \
	  echo "Executable not found; open the app and check ~/Library/Logs/WordHuntSolver.log"; \
	fi
