# Numerical Experiments

This directory contains code to reproduce the numerical experiments described
in the manuscript.

This code is developed with Julia version 1.10.10. To reproduce the
results, start Julia in this directory and execute the following commands in
the Julia REPL to create the figures shown in the paper.

```julia
julia> include("code.jl")

julia> kinetic_energy_comparison()

julia> semidiscrete_conservation()

julia> convergence_in_space() # this may take several minutes

julia> dispersive_shock_wave()

julia> fully_discrete_conservation()

julia> convergence_in_time() # this may take a few minutes

julia> error_growth()

julia> error_growth_gray_soliton()

julia> performance_comparison() # this may take half an hour

julia> fully_discrete_conservation_hyperbolization()

```

For the performance comparison with the work of
[Bai et. al. (2024)](https://doi.org/10.1137/22M152178X), to
perform the simulation with our code:
```julia
julia> run_single_simulation(relaxation=FullRelaxation(),dt=1/2048)
```
To perform the simulation with their code, clone the repository at
https://github.com/jiashhu/ME-Conserved-NLS and install all the required
dependencies.  Then:
```
cd ME-Conserved-NLS/Numerical_Tests
python3 NLS_Collo_1d.Main(1,40,1024,512,1,2,3,"Err","Standard_Soliton",20,"test",True)
```
The L_2 error information is sent to a file, but it can also be printed to the screen
by adding this line to the script `NLS_Collo_1d.py` at line 88:
```
print(myObj.endt_L2_ex_err_set[-1].real)
```
