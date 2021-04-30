# Rotating magnetohydrodynamic flow solver
The repository is meant to act as a source code control and back up of the project.

[![CI](https://github.com/j507/RotatingMHD/actions/workflows/main.yml/badge.svg)](https://github.com/j507/RotatingMHD/actions/workflows/main.yml)

Milestones:
- [x] Incompressible Navier-Stokes equations
- [x] Convective flow (Incompressible Navier-Stokes equations with bouyancy term + heat equation)
- [x] Rotational convective flow (Incompressible Navier-Stokes equations with bouyancy term and coriolis term + heat equation)
- [ ] Rotational magnetohydrodynamic flow (Incompressible Navier-Stokes equations with bouyancy term and coriolis term + heat equation + induction equation )

which are to be tested with the following benchmarks
1. Taylor-Green Vortex: https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex, Schaefer et. al (1996) and Guermond (2006) test
1. Christon, Gresho und Sutton (2002)
1. Christensen et. al (2000), Case 0
1. Christensen et. al (2000), Case 1 and Jackson et. al (2014)

To do list
- [x] Removing the bug in the adaptive mesh refinement branch
- [x] High Peclet number test and interpretation of the result
- [x] Remove hard-coded absolute tolerance
- [x] Adding the buoyancy to the Navier-Stokes solver
- [x] Interfacing of the entities and the solvers
- [x] Restructure the assembly scratch and copy structs and instance the FEValues in the Navier-Stokes solver with a Mapping
- [x] Reduce global communication and do not compute extrapolated values using vectors
- [x] Neumann boundary conditions in the incremental pressure projection scheme (Navier-Stokes solver)
- [x] Restructure the parameters of the solvers
- [x] Algebraic multigrid preconditioning in both solvers
- [ ] Python or bash script for running convergence tests
- [ ] Adaptive timestepping
- [ ] Initialization from analytical solution
- [ ] Restart from numerical solution



Questions
- [ ] High Reynolds number tests and inspection of the solution w.r.t. instabilities
- [ ] Stabilizitation of convectivion dominated flows?!
- [ ] Assessment of the performance in the MIT benchmark

