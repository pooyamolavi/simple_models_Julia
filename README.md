# simple_models_Julia
Julia codes for "Simple Models and Biased Forecasts," by Pooya Molavi (2022)

The code generates Figures 1-5 and C.1 in the paper. It also estimates the parameters of the NK model using data from https://fred.stlouisfed.org. The code also generates other numerical results of Sections 5-7. 

"NK.jl" generates the results of Section 5. It calls "NK_functions.jl" and "multi_variate_functions.jl" and uses the data in "GDPC1.xlsx," "GDPPOT.xlsx," "GDPDEF.xlsx," and "FEDFUNDS.xlsx." The file "NK_functions.jl" contains functions that are used in estimation of the NK model and compute the impulse response functions and the response to forward guidance.

"RBC.jl" generates the results of Section 6. It calls "RBC_functions.jl" and "multi_variate_functions.jl." It uses version 0.2 of the "SolveDSGE" Julia package. Newer versions of "SolveDSGE" are currently not supported. The file "RBC_functions.jl" contains functions that are used to compute the impulse response functions.

"DMP.jl" generates the results of Section 7. It calls "DMP_functions.jl" and "multi_variate_functions.jl." The file "DMP_functions.jl" contains functions that are used to compute the impulse response functions.

The file "multi_variate_functions.jl" contains a set of functions that compute pseudo-true 1-state models for various specifications of the true data-generating process. It is used by all other files.

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
