/*
 * solve_diffusion_system.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace Step35
{

 template <int dim>
 void NavierStokesProjection<dim>::projection_step(const bool reinit_prec)
 {
   pres_iterative.copy_from(pres_Laplace);

   pres_tmp = 0.;
   //for (unsigned d = 0; d < dim; ++d)
   pres_Diff.Tvmult_add(pres_tmp, u_n);

   phi_n_minus_1 = phi_n;

   static std::map<types::global_dof_index, double> bval;
   if (reinit_prec)
     VectorTools::interpolate_boundary_values(dof_handler_pressure,
                                              3,
                                              Functions::ZeroFunction<dim>(),
                                              bval);

   MatrixTools::apply_boundary_values(bval, pres_iterative, phi_n, pres_tmp);

   if (reinit_prec)
     prec_pres_Laplace.initialize(pres_iterative,
                                  SparseILU<double>::AdditionalData(
                                      vel_diag_strength, vel_off_diagonals));

   SolverControl solvercontrol(vel_max_its, vel_eps * pres_tmp.l2_norm());
   SolverCG<>    cg(solvercontrol);
   cg.solve(pres_iterative, phi_n, pres_tmp, prec_pres_Laplace);

   phi_n *= 1.5 / dt;
 }

template <int dim>
void NavierStokesProjection<dim>::update_pressure(const bool reinit_prec)
{
  pres_n_minus_1 = pres_n;
  switch (type)
  {
    case RunTimeParameters::Method::standard:
      pres_n += phi_n;
      break;
    case RunTimeParameters::Method::rotational:
      if (reinit_prec)
        prec_mass.initialize(pres_Mass);
      pres_n = pres_tmp;
      prec_mass.solve(pres_n);
      pres_n.sadd(1. / Re, 1., pres_n_minus_1);
      pres_n += phi_n;
      break;
    default:
      Assert(false, ExcNotImplemented());
  };
}

}  // namespace Step35

// explicit instantiations

template void Step35::NavierStokesProjection<2>::projection_step(const bool);
template void Step35::NavierStokesProjection<3>::projection_step(const bool);

template void Step35::NavierStokesProjection<2>::update_pressure(const bool);
template void Step35::NavierStokesProjection<3>::update_pressure(const bool);
