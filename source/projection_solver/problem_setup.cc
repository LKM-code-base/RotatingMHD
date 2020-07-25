#include <rotatingMHD/projection_solver.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>

namespace Step35
{
  template <int dim>
  void NavierStokesProjection<dim>::
  initialize()
  {
    setup_v_matrices();
    setup_p_matrices();
    setup_p_gradient_matrix();

    p_n.reinit(p_dof_handler.n_dofs());
    p_n_m1.reinit(p_dof_handler.n_dofs());
    p_rhs.reinit(p_dof_handler.n_dofs());
    p_tmp.reinit(p_dof_handler.n_dofs());
    phi_n.reinit(p_dof_handler.n_dofs());
    phi_n_m1.reinit(p_dof_handler.n_dofs());
    v_n.reinit(v_dof_handler.n_dofs());
    v_n_m1.reinit(v_dof_handler.n_dofs());
    v_extrapolated.reinit(v_dof_handler.n_dofs());
    v_rhs.reinit(v_dof_handler.n_dofs());
    v_tmp.reinit(v_dof_handler.n_dofs());

    assemble_v_matrices();
    assemble_p_matrices();
    assemble_p_gradient_matrix();

    if (!flag_adpative_time_step)
    {
      v_mass_plus_laplace_matrix = 0.;
      v_mass_plus_laplace_matrix.add(1.0 / Re, v_laplace_matrix);
      v_mass_plus_laplace_matrix.add(1.5 / dt_n, v_mass_matrix);
    }

    phi_n     = 0.;
    phi_n_m1  = 0.;

    p_initial_conditions.set_time(t_0);    
    VectorTools::interpolate(p_dof_handler, 
                             p_initial_conditions, 
                             p_n_m1);
    p_initial_conditions.advance_time(dt_n);
    VectorTools::interpolate(p_dof_handler, 
                             p_initial_conditions, 
                             p_n);

    v_initial_conditions.set_time(t_0);    
    VectorTools::interpolate(v_dof_handler,
                             v_initial_conditions,
                             v_n_m1);
    v_initial_conditions.advance_time(dt_n);
    VectorTools::interpolate(v_dof_handler,
                             v_initial_conditions,
                             v_n);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  setup_v_matrices()
  {
    {
      DynamicSparsityPattern dsp(v_dof_handler.n_dofs(),
                                 v_dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(v_dof_handler, 
                                      dsp,
                                      v_constraints);
      v_sparsity_pattern.copy_from(dsp);
      std::ofstream out ("v_sparsity_pattern.gpl");
      v_sparsity_pattern.print_gnuplot(out);
    }
    v_mass_plus_laplace_matrix.reinit(v_sparsity_pattern);
    v_system_matrix.reinit(v_sparsity_pattern);
    v_mass_matrix.reinit(v_sparsity_pattern);
    v_laplace_matrix.reinit(v_sparsity_pattern);
    v_advection_matrix.reinit(v_sparsity_pattern);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  setup_p_matrices()
  {
    {
      DynamicSparsityPattern dsp(p_dof_handler.n_dofs(),
                                 p_dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(p_dof_handler, 
                                      dsp,
                                      p_constraints);
      p_sparsity_pattern.copy_from(dsp);
      std::ofstream out ("p_sparsity_pattern.gpl");
      p_sparsity_pattern.print_gnuplot(out);
    }

    p_laplace_matrix.reinit(p_sparsity_pattern);
    p_system_matrix.reinit(p_sparsity_pattern);
    p_mass_matrix.reinit(p_sparsity_pattern);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  setup_p_gradient_matrix()
  {
    {
      DynamicSparsityPattern dsp(v_dof_handler.n_dofs(),
                                 p_dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(v_dof_handler,
                                      p_dof_handler,
                                      dsp);
      mixed_sparsity_pattern.copy_from(dsp);
      std::ofstream out("mixed_sparsity_pattern.gpl");
      mixed_sparsity_pattern.print_gnuplot(out);
    }
    p_gradient_matrix.reinit(mixed_sparsity_pattern);
  }
}

template void Step35::NavierStokesProjection<2>::initialize();
template void Step35::NavierStokesProjection<3>::initialize();
template void Step35::NavierStokesProjection<2>::setup_v_matrices();
template void Step35::NavierStokesProjection<3>::setup_v_matrices();
template void Step35::NavierStokesProjection<2>::setup_p_matrices();
template void Step35::NavierStokesProjection<3>::setup_p_matrices();
template void Step35::NavierStokesProjection<2>::setup_p_gradient_matrix();
template void Step35::NavierStokesProjection<3>::setup_p_gradient_matrix();