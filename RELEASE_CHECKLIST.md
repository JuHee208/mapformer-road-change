# Public Release Checklist

Use this checklist before pushing this repository to a public GitHub repo.

## 1. Legal and attribution

- Confirm whether the upstream `mxbh/mapformer` code can be redistributed as-is.
- Check the licenses and attribution requirements of MMSegmentation and Open-CD dependencies.
- Add a `LICENSE` file only after you are sure the redistribution terms are compatible with your fork.

## 2. Repository contents

- Keep datasets out of Git: `data/`
- Keep pretrained weights and experiment checkpoints out of Git: `model_ckpt/`, `runs/`
- Keep inference outputs and previews out of Git: `outputs/`, `tmp_*`
- Review `README.md` and `.gitignore` before the first public push

## 3. Final local review

Run:

```bash
cd /home/user/storage/c_hdd/JuHee/MapFormer/mapformer
git status
git diff -- README.md .gitignore RELEASE_CHECKLIST.md
```

## 4. Create the public GitHub repo

Recommended approach: keep the current remotes untouched and add a new public remote.

Example:

```bash
git remote add public https://github.com/JuHee208/<new-repo-name>.git
git push -u public HEAD:main
```

If `public` already exists:

```bash
git remote set-url public https://github.com/JuHee208/<new-repo-name>.git
git push -u public HEAD:main
```

## 5. Commit the publish-prep changes

Example:

```bash
git add README.md .gitignore RELEASE_CHECKLIST.md
git commit -m "Prepare public road change detection repository"
```

## 6. After the first push

- Re-open the GitHub repository page and verify that no dataset, checkpoint, or preview artifacts were uploaded.
- Check that the README renders correctly.
- Add repository topics such as `remote-sensing`, `change-detection`, `mapformer`, `semantic-change-detection`, and `road-extraction` if they fit.
