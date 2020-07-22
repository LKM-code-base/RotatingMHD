#include <rotatingMHD/projection_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

namespace Step35
{
  template <int dim>
  void NavierStokesProjection<dim>::
  correction_step(const bool reinit_prec)
  {
    /* Update for the next time step */
    p_n_m1 = p_n;
    switch (projection_method)
      {
        case RunTimeParameters::ProjectionMethod::standard:
          p_n += phi_n;
          break;
        case RunTimeParameters::ProjectionMethod::rotational:
          if (reinit_prec)
            p_update_preconditioner.initialize(p_mass_matrix);
          p_n = p_rhs;
          p_update_preconditioner.solve(p_n);
          p_n.sadd(1. / Re, 1., p_n_m1);
          p_n += phi_n;
          break;
        default:
          Assert(false, ExcNotImplemented());
      };
  }
}

template void Step35::NavierStokesProjection<2>::correction_step(const bool);
template void Step35::NavierStokesProjection<3>::correction_step(const bool);
