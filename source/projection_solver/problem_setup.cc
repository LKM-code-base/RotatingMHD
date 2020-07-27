#include <rotatingMHD/projection_solver.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>

namespace Step35
{
template <int dim>
void NavierStokesProjection<dim>::
setup_matrices_and_vectors()
{
  setup_velocity_matrices();
  setup_pressure_matrices();
  setup_pressure_gradient_matrix();

  pressure_n.reinit(pressure_dof_handler.n_dofs());
  pressure_n_minus_1.reinit(pressure_dof_handler.n_dofs());
  pressure_rhs.reinit(pressure_dof_handler.n_dofs());
  pressure_tmp.reinit(pressure_dof_handler.n_dofs());
  phi_n.reinit(pressure_dof_handler.n_dofs());
  phi_n_minus_1.reinit(pressure_dof_handler.n_dofs());
  velocity_n.reinit(velocity_dof_handler.n_dofs());
  velocity_n_minus_1.reinit(velocity_dof_handler.n_dofs());
  extrapolated_velocity.reinit(velocity_dof_handler.n_dofs());
  velocity_rhs.reinit(velocity_dof_handler.n_dofs());
  velocity_tmp.reinit(velocity_dof_handler.n_dofs());
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_constant_matrices()
{
  assemble_velocity_matrices();
  assemble_pressure_matrices();
  assemble_pressure_gradient_matrix();

  if (!flag_adpative_time_step)
  {
    velocity_mass_plus_laplace_matrix = 0.;
    velocity_mass_plus_laplace_matrix.add(1.0 / Re, 
                                          velocity_laplace_matrix);
    velocity_mass_plus_laplace_matrix.add(1.5 / dt_n, 
                                          velocity_mass_matrix);
  }
}

template <int dim>
void NavierStokesProjection<dim>::
setup_velocity_matrices()
{
  {
    DynamicSparsityPattern dsp(velocity_dof_handler.n_dofs(),
                                velocity_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(velocity_dof_handler, 
                                    dsp,
                                    velocity_constraints);
    velocity_sparsity_pattern.copy_from(dsp);
    std::ofstream out ("velocity_sparsity_pattern.gpl");
    velocity_sparsity_pattern.print_gnuplot(out);
  }
  velocity_mass_plus_laplace_matrix.reinit(velocity_sparsity_pattern);
  velocity_system_matrix.reinit(velocity_sparsity_pattern);
  velocity_mass_matrix.reinit(velocity_sparsity_pattern);
  velocity_laplace_matrix.reinit(velocity_sparsity_pattern);
  velocity_advection_matrix.reinit(velocity_sparsity_pattern);
}

template <int dim>
void NavierStokesProjection<dim>::
setup_pressure_matrices()
{
  {
    DynamicSparsityPattern dsp(pressure_dof_handler.n_dofs(),
                                pressure_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(pressure_dof_handler, 
                                    dsp,
                                    pressure_constraints);
    pressure_sparsity_pattern.copy_from(dsp);
    std::ofstream out ("pressure_sparsity_pattern.gpl");
    pressure_sparsity_pattern.print_gnuplot(out);
  }

  pressure_laplace_matrix.reinit(pressure_sparsity_pattern);
  //pressure_system_matrix.reinit(pressure_sparsity_pattern);
  pressure_mass_matrix.reinit(pressure_sparsity_pattern);
}

template <int dim>
void NavierStokesProjection<dim>::
setup_pressure_gradient_matrix()
{
  {
    DynamicSparsityPattern dsp(velocity_dof_handler.n_dofs(),
                                pressure_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(velocity_dof_handler,
                                    pressure_dof_handler,
                                    dsp);
    mixed_sparsity_pattern.copy_from(dsp);
    std::ofstream out("mixed_sparsity_pattern.gpl");
    mixed_sparsity_pattern.print_gnuplot(out);
  }
  pressure_gradient_matrix.reinit(mixed_sparsity_pattern);
}

template <int dim>
void NavierStokesProjection<dim>::
initialize()
{
  phi_n         = 0.;
  phi_n_minus_1 = 0.;

  pressure_initial_conditions.set_time(t_0);    
  VectorTools::interpolate(pressure_dof_handler, 
                            pressure_initial_conditions, 
                            pressure_n_minus_1);
  pressure_initial_conditions.advance_time(dt_n);
  VectorTools::interpolate(pressure_dof_handler, 
                            pressure_initial_conditions, 
                            pressure_n);

  velocity_initial_conditions.set_time(t_0);    
  VectorTools::interpolate(velocity_dof_handler,
                            velocity_initial_conditions,
                            velocity_n_minus_1);
  velocity_initial_conditions.advance_time(dt_n);
  VectorTools::interpolate(velocity_dof_handler,
                            velocity_initial_conditions,
                            velocity_n);
}

}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::setup_matrices_and_vectors();
template void Step35::NavierStokesProjection<3>::setup_matrices_and_vectors();
template void Step35::NavierStokesProjection<2>::assemble_constant_matrices();
template void Step35::NavierStokesProjection<3>::assemble_constant_matrices();
template void Step35::NavierStokesProjection<2>::setup_velocity_matrices();
template void Step35::NavierStokesProjection<3>::setup_velocity_matrices();
template void Step35::NavierStokesProjection<2>::setup_pressure_matrices();
template void Step35::NavierStokesProjection<3>::setup_pressure_matrices();
template void Step35::NavierStokesProjection<2>::setup_pressure_gradient_matrix();
template void Step35::NavierStokesProjection<3>::setup_pressure_gradient_matrix();
template void Step35::NavierStokesProjection<2>::initialize();
template void Step35::NavierStokesProjection<3>::initialize();