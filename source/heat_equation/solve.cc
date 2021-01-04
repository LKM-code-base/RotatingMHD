#include <rotatingMHD/heat_equation.h>

#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void HeatEquation<dim>::solve()
{
  if (temperature->solution.size() != mass_matrix.m())
  {
    setup();
    flag_reinit_preconditioner            = true;
    flag_add_mass_and_stiffness_matrices  = true;
  }

  assemble_linear_system();

  rhs_norm = rhs.l2_norm();

  solve_linear_system(flag_reinit_preconditioner ||
                      time_stepping.get_step_number() %
                      parameters.solver_update_preconditioner == 0);
}

template <int dim>
void HeatEquation<dim>::assemble_linear_system()
{
  // System matrix setup
  if (time_stepping.coefficients_changed() == true ||
      flag_add_mass_and_stiffness_matrices)
  {
      TimerOutput::Scope  t(*computing_timer, "Heat Equation: Matrix summation");

    mass_plus_stiffness_matrix = 0.;

    mass_plus_stiffness_matrix.add(
      time_stepping.get_alpha()[0] / time_stepping.get_next_step_size(),
      mass_matrix);

    mass_plus_stiffness_matrix.add(
      time_stepping.get_gamma()[0] / parameters.Pe,
      stiffness_matrix);

      flag_add_mass_and_stiffness_matrices = false;
  }

  if (parameters.flag_semi_implicit_convection &&
      !flag_ignore_advection)
  {
    assemble_advection_matrix();
    system_matrix.copy_from(mass_plus_stiffness_matrix);
    system_matrix.add(1.0, advection_matrix);
  }
  
  // Right hand side setup
  assemble_rhs();
}

template <int dim>
void HeatEquation<dim>::solve_linear_system(const bool reinit_preconditioner)
{
  if (parameters.verbose)
  *pcout << "  Heat Equation: Solving linear system...";

  TimerOutput::Scope  t(*computing_timer, "Heat Equation: Solve");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_temperature(rhs);
  distributed_temperature = temperature->solution;

  /* The following pointer holds the address to the correct matrix 
  depending on if the semi-implicit scheme is chosen or not */
  const LinearAlgebra::MPI::SparseMatrix  *system_matrix_ptr;
  if (parameters.flag_semi_implicit_convection && 
      !flag_ignore_advection)
    system_matrix_ptr = &system_matrix;
  else
    system_matrix_ptr = &mass_plus_stiffness_matrix;

  if (reinit_preconditioner)
    preconditioner.initialize(*system_matrix_ptr);

  SolverControl solver_control(parameters.n_maximum_iterations,
                               std::max(parameters.relative_tolerance * rhs_norm,
                                        absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverGMRES solver(solver_control,
                                      mpi_communicator);
  #else
    LinearAlgebra::SolverGMRES solver(solver_control);
  #endif

  try
  {
    solver.solve(*system_matrix_ptr,
                 distributed_temperature,
                 rhs,
                 preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method of heat equation: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception in the solve method of the heat equation!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  temperature->constraints.distribute(distributed_temperature);

  temperature->solution = distributed_temperature;

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Number of GMRES iterations: " 
           << solver_control.last_step()
           << ", Final residual: " << solver_control.last_value() << "."
           << std::endl << std::endl;
}

} // namespace RMHD

// explicit instantiations
template void RMHD::HeatEquation<2>::solve();
template void RMHD::HeatEquation<3>::solve();

template void RMHD::HeatEquation<2>::assemble_linear_system();
template void RMHD::HeatEquation<3>::assemble_linear_system();

template void RMHD::HeatEquation<2>::solve_linear_system(const bool);
template void RMHD::HeatEquation<3>::solve_linear_system(const bool);
