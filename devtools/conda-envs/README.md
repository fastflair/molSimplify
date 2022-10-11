For M1 Mac users: If the current `mols_m1.yml` file is not working for you, try using the following lines in `mols_m1.yml` instead:

```
name: mols_test_m1
channels:
     - hjkgroup
     - conda-forge
     - anaconda
     - theochem
     - defaults
dependencies:
     - python=3.8
     - openbabel
     - tensorflow-base
     - pyqt
     - pip
     - qt
     - scikit-learn
     - pandas
     - scipy
     - pymongo
     - libffi
```

Some caveats for this approach:
- Uses Openbabel 3 which means some of the tests fail because of the different connecting atoms definition.
- GUI does not work well (unusable on dark mode setting).
