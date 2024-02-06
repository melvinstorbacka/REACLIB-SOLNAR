# REACLIB-SOLNAR

**REACLIB** **S**tockholm **O**pen **L**ibrary for **N**uclear **A**strophysical **R**eaction Rate Variations


**Data generation**
For data generation, TALYS 2.0 is used ([https://www-nds.iaea.org/talys/](https://www-nds.iaea.org/talys/)). 
To obtain a more detailed temperature parameter space, the source code of TALYS was slightly adjusted.

<pre>
  1. ./A0_talys_mod.f90:80:  integer, parameter :: numT=<b>108</b>           ! number of temperatures
  2. ./egridastro.f90:111:  if (Teps > 1.) dTgrid = <b>0.10</b>  
  3. ./egridastro.f90:112:  if (Teps > 4.) dTgrid = <b>0.10</b>  
</pre>

