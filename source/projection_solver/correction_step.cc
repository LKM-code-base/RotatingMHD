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
        static TrilinosWrappers::SolverDirect::AdditionalData data(
                                                          false, 
                                                          "Amesos_Klu");
        static SolverControl solver_control(1, 0);
        {
          TrilinosWrappers::MPI::Vector distributed_pressure_n(pressure_rhs);
          TrilinosWrappers::MPI::Vector distributed_pressure_n_minus_1(pressure_rhs);
          TrilinosWrappers::MPI::Vector distributed_phi_n(pressure_rhs);

          distributed_pressure_n = pressure_n;
          distributed_pressure_n_minus_1 = pressure_n_minus_1;
          distributed_phi_n = phi_n;

          /* Using a direct solver */
          TrilinosWrappers::SolverDirect solver(solver_control, data);
          solver.solve(pressure_mass_matrix,
                       distributed_pressure_n, 
                       pressure_rhs);

          /* Using CG */
          /*if (reinit_prec)
            correction_step_preconditioner.initialize(
                                              pressure_mass_matrix);

          SolverControl solver__control(solver_max_iterations, 
                                solver_tolerance * pressure_rhs.l2_norm());
          
          SolverCG<TrilinosWrappers::MPI::Vector>    cg_solver(solver__control);
          cg_solver.solve(pressure_mass_matrix, 
                  distributed_pressure_n, 
                  pressure_rhs, 
                  correction_step_preconditioner);*/

          pressure_constraints.distribute(distributed_pressure_n);
          distributed_pressure_n.sadd(1.0 / Re, 1., distributed_pressure_n_minus_1);
          distributed_pressure_n += distributed_phi_n;
          pressure_n = distributed_pressure_n;
        }

        break;
      default:
        Assert(false, ExcNotImplemented());
    };
}
}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::pressure_correction(const bool);
template void Step35::NavierStokesProjection<3>::pressure_correction(const bool);
