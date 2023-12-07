# REACLIB-SOLNAR

**REACLIB** **S**tockholm **O**pen **L**ibrary for **N**uclear **A**strophysical **R**eaction Rate Variations


**Data generation**
For data generation, TALYS 1.96 is used (https://tendl.web.psi.ch/tendl_2021/talys.html). 
To obtain a more detailed temperature parameter space, the source code of TALYS was slightly adjusted.

1. talys/source/talys.cmb: line 101: numT = 108
2. talys/source/egridastro.f: line 50: dTgrid=0.1
3. talys/source/egridastro.f: line 51: dTgrid=0.1
