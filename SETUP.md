# Local Setup Guide

Use this setup after cloning the repository on a new computer.

## 1. Install Python

Install Python 3.12 or a compatible Python 3 version, then confirm it works:

```powershell
python --version
```

## 2. Create A Virtual Environment

Do not copy or commit `.venv`. Create it locally after cloning:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run The System

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## 5. Common Clone Issues

If the cloned project does not run, check these first:

- The virtual environment was not created yet.
- Dependencies were not installed from `requirements.txt`.
- The local database in `instance/clinic.db` is missing and must be created by running the app.
- Upload/test files are local data and may not be included in the repository.
- Port `5000` is already being used by another Flask process.

## 6. Git Notes

Virtual environment folders are machine-specific and should stay ignored:

```text
.venv/
venv/
.venv_broken/
venv_broken/
```

If these were previously committed, remove them from Git tracking with:

```powershell
git rm --cached -r .venv venv .venv_broken venv_broken
git add .gitignore SETUP.md
git commit -m "Ignore local virtual environments"
```

If Git reports `index.lock`, close VS Code, stop any running `git.exe` process, delete `.git/index.lock`, then retry the Git command.
