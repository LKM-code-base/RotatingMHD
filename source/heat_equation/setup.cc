#include <rotatingMHD/heat_equation.h>

#include <deal.II/dofs/dof_tools.h>
#ifdef USE_PETSC_LA
  #include <deal.II/lac/dynamic_sparsity_pattern.h>
  #include <deal.II/lac/sparsity_tools.h>
#else
  #include <deal.II/lac/trilinos_sparsity_pattern.h>
#endif
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void HeatEquation<dim>::setup()
{
  setup_matrices();

  setup_vectors();

  assemble_constant_matrices();
}

template <int dim>
void HeatEquation<dim>::setup_matrices()
{
  if (parameters.verbose)
    *pcout << "  Heat Equation: Setting up matrices...";

  TimerOutput::Scope  t(*computing_timer, "Heat Equation: Setup - Matrices");

  mass_matrix.clear();
  stiffness_matrix.clear();
  advection_matrix.clear();
  system_matrix.clear();

  {
    #ifdef USE_PETSC_LA
      DynamicSparsityPattern
      sparsity_pattern(temperature->locally_relevant_dofs);

      DoFTools::make_sparsity_pattern(
        *temperature->dof_handler,
        sparsity_pattern,
        temperature->constraints,
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator));

      SparsityTools::distribute_sparsity_pattern
      (sparsity_pattern,
       temperature->locally_owned_dofs,
       mpi_communicator,
       temperature->locally_relevant_dofs);

      mass_matrix.reinit
      (temperature->locally_owned_dofs,
       temperature->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      stiffness_matrix.reinit
      (temperature->locally_owned_dofs,
       temperature->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      mass_plus_stiffness_matrix.reinit
      (temperature->locally_owned_dofs,
       temperature->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      advection_matrix.reinit
      (temperature->locally_owned_dofs,
       temperature->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      system_matrix.reinit
      (temperature->locally_owned_dofs,
       temperature->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);

    #else
      TrilinosWrappers::SparsityPattern
      sparsity_pattern(temperature->locally_owned_dofs,
                       temperature->locally_owned_dofs,
                       temperature->locally_relevant_dofs,
                       mpi_communicator);

      DoFTools::make_sparsity_pattern(
        *temperature->dof_handler,
        sparsity_pattern,
        temperature->constraints,
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator));

      sparsity_pattern.compress();

      mass_matrix.reinit(sparsity_pattern);
      stiffness_matrix.reinit(sparsity_pattern);
      mass_plus_stiffness_matrix.reinit(sparsity_pattern);
      advection_matrix.reinit(sparsity_pattern);
      system_matrix.reinit(sparsity_pattern);

    #endif
  }
  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

template <int dim>
void HeatEquation<dim>::
setup_vectors()
{
  if (parameters.verbose)
    *pcout << "  Heat Equation: Setting up vectors...";

  TimerOutput::Scope  t(*computing_timer, "Heat Equation: Setup - Vectors");

  // Initializing the temperature related vectors
  rhs.reinit(temperature->distributed_vector);
  temperature_tmp.reinit(temperature->solution);

  // Initializing the velocity related vector
  if (!flag_ignore_advection)
    extrapolated_velocity.reinit(velocity->solution);
  
  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

template <int dim>
void HeatEquation<dim>::set_source_term(
  Function<dim> &source_term)
{
  source_term_ptr = &source_term;
}

} // namespace RMHD

// explicit instantiations
template void RMHD::HeatEquation<2>::setup();
template void RMHD::HeatEquation<3>::setup();

template void RMHD::HeatEquation<2>::setup_matrices();
template void RMHD::HeatEquation<3>::setup_matrices();

template void RMHD::HeatEquation<2>::setup_vectors();
template void RMHD::HeatEquation<3>::setup_vectors();

template void RMHD::HeatEquation<2>::set_source_term(Function<2> &);
template void RMHD::HeatEquation<3>::set_source_term(Function<3> &);