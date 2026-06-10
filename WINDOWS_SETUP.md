# Windows setup

## Quick start

1. Install Python 3.10 or newer from https://www.python.org/downloads/windows/.
2. During install, enable "Add python.exe to PATH".
3. Copy this whole project folder to the other Windows PC.
4. Double-click `run_gui.bat` in the project root.
5. Open http://127.0.0.1:8080/ if the browser does not open automatically.

The launcher creates a project-local `.venv` folder and installs `requirements.txt`.
Internal crawler/tagger subprocesses use the same Python runtime, so the folder can
live under any Windows user path.

## Optional full NLP runtime

The default launcher enables keyword sentiment fallback when heavy model packages
are not installed. To use FinBERT/BGE/Hugging Face local model features, run:

```bat
.\.venv\Scripts\python.exe -m pip install -r requirements-nlp.txt
```

Model packages are large and may take a while to install.
