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

  set_preconditioner_data();
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

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void HeatEquation<dim>::set_source_term(
  Function<dim> &source_term)
{
  source_term_ptr = &source_term;
}



template <int dim>
void HeatEquation<dim>::set_preconditioner_data()
{
  #ifdef USE_PETSC_LA
    if (parameters.convective_term_time_discretization ==
          RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit)
      amg_data.symmetric_operator               = true;
    if (dim == 2)
      amg_data.strong_threshold               = 0.25;
    else if (dim == 3)
      amg_data.strong_threshold               = 0.50;
    amg_data.max_row_sum                      = 0.9;
    amg_data.aggressive_coarsening_num_levels = 0;
    amg_data.output_details                   = false;
  #else
    FEValuesExtractors::Scalar                scalar_extractor(0);
    DoFTools::extract_constant_modes(
      *temperature->dof_handler,
      (*temperature->dof_handler).get_fe_collection().component_mask(scalar_extractor),
      amg_data.constant_modes );

    amg_data.elliptic                         = false;
    if (temperature->fe_degree > 1)
      amg_data.higher_order_elements          = true;
    amg_data.n_cycles                         = 1;
    amg_data.w_cycle                          = false;
    amg_data.aggregation_threshold            = 1e-4;
    amg_data.smoother_sweeps                  = 2;
    amg_data.smoother_overlap                 = 0;
    amg_data.output_details                   = false;
    amg_data.smoother_type                    = "Chebyshev";
    amg_data.coarse_type                      = "Amesos-KLU";
  #endif
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

template void RMHD::HeatEquation<2>::set_preconditioner_data();
template void RMHD::HeatEquation<3>::set_preconditioner_data();