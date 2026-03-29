# DL-course

Deep learning course experiments managed in a single repository.

## Structure

- `docs/`: course handouts or experiment notes
- `prj_1/`: experiment 1, image classification
- `prj_2/`: experiment 2 placeholder
- `shared/`: reusable utilities across experiments
- `scripts/`: helper scripts for setup or sync
- `env/`: environment notes or exported dependency files

## Suggested Workflow

1. Write and test code locally.
2. Commit and push to GitHub.
3. Pull the latest changes on the server.
4. Run the target experiment directory on the server.

## Quick Start

```bash
git init
git add .
git commit -m "Initialize DL course repository"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

On the server:

```bash
git clone <your-github-repo-url>
cd DL-course
git pull
cd prj_1
python train.py
```
