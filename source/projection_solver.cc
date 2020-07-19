/*
 * projection_solver.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>

namespace Step35
{

template <int dim>
NavierStokesProjection<dim>::NavierStokesProjection(
const RunTimeParameters::Data_Storage &data)
:
type(data.form),
deg(data.pressure_degree),
dt(data.dt),
t_0(data.initial_time),
T(data.final_time),
Re(data.Reynolds),
vel_exact(data.initial_time),
fe_velocity(FE_Q<dim>(deg + 1), dim),
fe_pressure(deg),
dof_handler_velocity(triangulation),
dof_handler_pressure(triangulation),
quadrature_pressure(deg + 1),
quadrature_velocity(deg + 2),
vel_max_its(data.vel_max_iterations),
vel_Krylov_size(data.vel_Krylov_size),
vel_off_diagonals(data.vel_off_diagonals),
vel_update_prec(data.vel_update_prec),
vel_eps(data.vel_eps),
vel_diag_strength(data.vel_diag_strength)
{
  if (deg < 1)
    std::cout << " WARNING: The chosen pair of finite element spaces is not stable."
              << std::endl
              << " The obtained results will be nonsense" << std::endl;

  AssertThrow(!((dt <= 0.) || (dt > .5 * T)), ExcInvalidTimeStep(dt, .5 * T));

  create_triangulation_and_dofs(data.n_global_refines);

  initialize();
}


template <int dim>
void NavierStokesProjection<dim>::initialize()
{
  vel_Laplace_plus_Mass = 0.;
  vel_Laplace_plus_Mass.add(1. / Re, vel_Laplace);
  vel_Laplace_plus_Mass.add(1.5 / dt, vel_Mass);

  EquationData::PressureIC<dim> pres(t_0);
  VectorTools::interpolate(dof_handler_pressure, pres, pres_n_minus_1);
  pres.advance_time(dt);
  VectorTools::interpolate(dof_handler_pressure, pres, pres_n);
  phi_n         = 0.;
  phi_n_minus_1 = 0.;
  //for (unsigned int d = 0; d < dim; ++d)
  //  {
  vel_exact.set_time(t_0);
  VectorTools::interpolate(dof_handler_velocity,
                           Functions::ZeroFunction<dim>(dim),
                           u_n_minus_1);
  vel_exact.advance_time(dt);
  VectorTools::interpolate(dof_handler_velocity,
                           Functions::ZeroFunction<dim>(dim),
                           u_n);
  //  }
}

template <int dim>
void NavierStokesProjection<dim>::run(const bool         verbose,
                                      const unsigned int output_interval)
                                      {
  ConditionalOStream verbose_cout(std::cout, verbose);

  const auto n_steps = static_cast<unsigned int>((T - t_0) / dt);
  vel_exact.set_time(2. * dt);
  output_results(1);
  for (unsigned int n = 2; n <= n_steps; ++n)
    {
      if (n % output_interval == 0)
        {
          verbose_cout << "Plotting Solution" << std::endl;
          output_results(n);
        }
      std::cout << "Step = " << n << " Time = " << (n * dt) << std::endl;
      verbose_cout << "  Interpolating the velocity " << std::endl;

      interpolate_velocity();
      verbose_cout << "  Diffusion Step" << std::endl;
      if (n % vel_update_prec == 0)
        verbose_cout << "    With reinitialization of the preconditioner"
        << std::endl;
      diffusion_step((n % vel_update_prec == 0) || (n == 2));

      verbose_cout << "  Projection Step" << std::endl;
      projection_step((n == 2));
      verbose_cout << "  Updating the Pressure" << std::endl;
      update_pressure((n == 2));
      vel_exact.advance_time(dt);
    }
  output_results(n_steps);
                                      }



template <int dim>
void NavierStokesProjection<dim>::interpolate_velocity()
{
  //for (unsigned int d = 0; d < dim; ++d)
  //  {

  u_star.equ(2., u_n);
  u_star -= u_n_minus_1;

  //  }
}

}  // namespace Step35

// explicit instantiations

template class Step35::NavierStokesProjection<2>;
template class Step35::NavierStokesProjection<3>;

