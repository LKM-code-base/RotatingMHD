#include <rotatingMHD/magnetic_induction.h>



namespace RMHD
{



namespace Solvers
{



template <int dim>
void MagneticInduction<dim>::initialization_step()
{
  if (this->ptr_supply_term != nullptr)
    this->ptr_supply_term->set_time(this->time_stepping.get_start_time());

  {
    if (parameters.verbose)
      *this->pcout << "  Magnetic Induction: Assembling the zeroth step's right hand side...";

    TimerOutput::Scope  t(*this->computing_timer, "Magnetic Induction: Zeroth step - RHS Assembly");

    assemble_initialization_step_rhs();

    if (parameters.verbose)
      *this->pcout << " done!" << std::endl
            << "    Right-hand side's L2-norm = "
            << std::scientific << std::setprecision(6)
            << this->norm_projection_step_rhs
            << std::endl;
  }

  {
    if (parameters.verbose)
      *this->pcout << "  Magnetic Induction: Solving the zeroth step...";

    TimerOutput::Scope  t(*this->computing_timer, "Magnetic Induction: Zeroth step - Solve");

    std::pair<int, double> feedback;

    feedback = this->solve_initialization_step();

    if (parameters.verbose)
      *this->pcout << " done!" << std::endl
            << "    Number of CG iterations: "
            << feedback.first
            << ", Final residual: "
            << feedback.second << "."
            << std::endl;
  }
}



template <int dim>
void MagneticInduction<dim>::diffusion_step(const bool reinit_preconditioner)
{
  assemble_diffusion_step();

  {
    if (parameters.verbose)
      *this->pcout << "  Magnetic Induction: Solving the diffusion step...";

    TimerOutput::Scope  t(*this->computing_timer, "Magnetic Induction: Diffusion step - Solve");

    std::pair<int, double> feedback;

    feedback = this->solve_diffusion_step(reinit_preconditioner);

    if (parameters.verbose)
      *this->pcout << " done!" << std::endl
            << "    Number of CG iterations: "
            << feedback.first
            << ", Final residual: "
            << feedback.second << "."
            << std::endl;
  }
}



template <int dim>
void MagneticInduction<dim>::projection_step(const bool reinit_preconditioner)
{
  {
    if (parameters.verbose)
      *this->pcout << "  Magnetic Induction: Assembling the projection step's right hand side...";

    TimerOutput::Scope  t(*this->computing_timer, "Magnetic Induction: Projection step - RHS Assembly");

    assemble_projection_step_rhs();

    if (parameters.verbose)
      *this->pcout << " done!" << std::endl
            << "    Right-hand side's L2-norm = "
            << std::scientific << std::setprecision(6)
            << this->norm_projection_step_rhs
            << std::endl;
  }

  {
    if (parameters.verbose)
      *this->pcout << "  Magnetic Induction: Solving the projection step...";

    TimerOutput::Scope  t(*this->computing_timer, "Magnetic Induction: Projection step - Solve");

    std::pair<int, double> feedback;

    feedback = this->solve_projection_step(reinit_preconditioner);

    if (parameters.verbose)
      *this->pcout << " done!" << std::endl
            << "    Number of CG iterations: "
            << feedback.first
            << ", Final residual: "
            << feedback.second << "."
            << std::endl;
  }
}



template <int dim>
void MagneticInduction<dim>::correction_step(const bool /* */)
{
  if (parameters.verbose)
    *this->pcout << "  Magnetic Induction: Correction step...";

  TimerOutput::Scope  t(*this->computing_timer, "Magnetic Induction: Correction step");

  LinearAlgebra::MPI::Vector distributed_old_pseudo_pressure(pseudo_pressure->distributed_vector);
  LinearAlgebra::MPI::Vector distributed_auxiliary_scalar(this->auxiliary_scalar->distributed_vector);

  distributed_old_pseudo_pressure = pseudo_pressure->old_solution;
  distributed_auxiliary_scalar    = this->auxiliary_scalar->solution;

  distributed_old_pseudo_pressure  += distributed_auxiliary_scalar;

  pseudo_pressure->solution = distributed_old_pseudo_pressure;

  if (this->flag_mean_value_constraint)
  {
    const LinearAlgebra::MPI::Vector::value_type mean_value
      = VectorTools::compute_mean_value(pseudo_pressure->get_dof_handler(),
                                        QGauss<dim>(pseudo_pressure->fe_degree() + 1),
                                        pseudo_pressure->solution,
                                        0);

    distributed_old_pseudo_pressure.add(-mean_value);
    pseudo_pressure->solution = distributed_old_pseudo_pressure;
  }

  if (parameters.verbose)
    *this->pcout << " done!" << std::endl << std::endl;
}



} // namespace Solvers



} // namespace RMHD



// Explicit instantiations
template void RMHD::Solvers::MagneticInduction<2>::initialization_step();
template void RMHD::Solvers::MagneticInduction<3>::initialization_step();

template void RMHD::Solvers::MagneticInduction<2>::diffusion_step(const bool);
template void RMHD::Solvers::MagneticInduction<3>::diffusion_step(const bool);

template void RMHD::Solvers::MagneticInduction<2>::projection_step(const bool);
template void RMHD::Solvers::MagneticInduction<3>::projection_step(const bool);

template void RMHD::Solvers::MagneticInduction<2>::correction_step(const bool);
template void RMHD::Solvers::MagneticInduction<3>::correction_step(const bool);
