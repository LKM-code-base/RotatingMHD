# Rotating magnetohydrodynamic flow solver
The repository is meant to act as a source code control and back up of the project.

Milestones:
- [ ] Incompressible Navier-Stokes equations
- [ ] Convective flow (Incompressible Navier-Stokes equations with bouyancy term + heat equation)
- [ ] Rotational convective flow (Incompressible Navier-Stokes equations with bouyancy term and coriolis term + heat equation)
- [ ] Rotational magnetohydrodynamic flow (Incompressible Navier-Stokes equations with bouyancy term and coriolis term + heat equation + induction equation )

which are to be tested with the following benchmarks
1. Taylor-Green Vortex: https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex, Schaefer et. al (1996) and Guermond (2006) test
1. Christon, Gresho und Sutton (2002)
1. Christensen et. al (2000), Case 0
1. Christensen et. al (2000), Case 1 and Jackson et. al (2014)

To do list
- [ ] Removing the bug in the adaptive mesh refinement branch
- [ ] Adaptive timestepping
- [ ] High Reynolds number tests and inspection of the solution w.r.t. instabilities
- [ ] High Peclet number test and interpretation of the result
- [ ] Neumann boundary conditions in the incremental pressure projection scheme (Navier-Stokes solver)
- [ ] Interfacing of the entities and the solvers
- [ ] Adding the buoyancy to the Navier-Stokes solver
- [ ] Algebraic multigrid preconditioning in both solvers
