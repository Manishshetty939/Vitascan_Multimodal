# VITASCAN

This repository contains code for vitamin deficiency classification and multimodal fusion models.

Important: large data, feature arrays, and trained models are intentionally excluded from the repo. Keep them locally or store them in an external storage service.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
# pip install -r requirements.txt (add if you maintain one)
```

2. Initialize and push repository to GitHub (see below).

Creating the remote repository

- Using GitHub CLI (recommended):

```bash
gh repo create <owner>/<repo-name> --public --source=. --remote=origin
git branch -M main
git push -u origin main
```

- Or create a repository on GitHub web and then:

```bash
git remote add origin https://github.com/<owner>/<repo-name>.git
git branch -M main
git push -u origin main
```

If you want help creating the remote repository from here, tell me whether you have `gh` installed and whether you want the repo public or private.
# Vitascan
