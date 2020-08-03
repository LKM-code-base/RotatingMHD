#include <rotatingMHD/projection_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

namespace Step35
{
template <int dim>
void NavierStokesProjection<dim>::
pressure_correction(const bool reinit_prec)
{
  /* Update for the next time step */
  pressure_n_minus_1 = pressure_n;
  switch (projection_method)
    {
      case RunTimeParameters::ProjectionMethod::standard:
        pressure_n += phi_n;
        break;
      case RunTimeParameters::ProjectionMethod::rotational:
        if (reinit_prec)
          pressure_correction_solver.initialize(pressure_mass_matrix);
        pressure_n = pressure_rhs;
        pressure_correction_solver.solve(pressure_n);
        pressure_n.sadd(1. / Re, 1., pressure_n_minus_1);
        pressure_n += phi_n;
        break;
      default:
        Assert(false, ExcNotImplemented());
    };
}
}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::pressure_correction(const bool);
template void Step35::NavierStokesProjection<3>::pressure_correction(const bool);
