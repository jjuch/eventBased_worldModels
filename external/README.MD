# The external folder
This folder can be used to store official stable LeWorldModel as given by https://github.com/lucas-maes/le-wm.git.

A fork exists on https://github.com/jjuch/le-wm.git, and more specifically on the branch `ball-temporal-jepa`.

## Install
To install the exact version of the clone, use:
```bash
git clone --recurse-submodules \
  https://github.com/jjuch/eventBased_worldModels.git
```
or
```bash
git submodule update --init --recursive
```
Afterwards, install the editable version of the submodule:
```bash
pip install -e external/le-wm
```
Note that you need to have `swig` installed and on your path.

Below I have added for reference how to add the submodule. But this is not necessary to clone this repository.

## Push update of external library
Push to the fork first:
```bash
cd external/le-wm

git switch ball-temporal-jepa
git add .
git commit -m "Package reusable LeWorldModel core"
git push origin ball-temporal-jepa
```

Then update the parent repository's pinned submodule commit:
```bash
cd ../..

git submodule update \
    --remote \
    external/le-wm

git add external/le-wm
git commit -m "Update pinned LeWorldModel dependency"
git push
```

## First install - for reference
### Add local clone
Configure the local clone:
```bash
git clone https://github.com/jjuch/le-wm.git external/le-wm
cd external/le-wm

git remote add upstream https://github.com/lucas-maes/le-wm.git
git fetch upstream
```

Regularly synchronize the clean branch:
```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```
The modification branch then derives from a specific upstream commit:
```bash
git switch ball-temporal-jepa
```

### Add as a submodule
From the root of `eventBased_worldModels:
```bash
mkdir -p external

git submodule add \
  -b ball-temporal-jepa \
  https://github.com/jjuch/le-wm.git \
  external/le-wm
```
and commit the submodule pointer:
```bash
git add .gitmodules external/le-wm
git commit -m "Add pinned LeWorldModel reference implementation"
```