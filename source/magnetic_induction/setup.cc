#include <rotatingMHD/magnetic_induction.h>



namespace RMHD
{



namespace Solvers
{



template <int dim>
void MagneticInduction<dim>::setup()
{
  if (this->flag_setup_auxiliary_scalar)
    setup_pseudo_pressure();

  ProjectionSolverBase<dim>::setup();
}



template <int dim>
void MagneticInduction<dim>::setup_pseudo_pressure()
{
  pseudo_pressure->setup_dofs();
  pseudo_pressure->setup_vectors();

  pseudo_pressure->clear_boundary_conditions();
  pseudo_pressure->setup_boundary_conditions();

  /* Set boundary conditions */

  pseudo_pressure->close_boundary_conditions();
  pseudo_pressure->apply_boundary_conditions();

  pseudo_pressure->set_solution_vectors_to_zero();
}



template <int dim>
void MagneticInduction<dim>::setup_auxiliary_scalar()
{
  this->auxiliary_scalar->setup_dofs();
  this->auxiliary_scalar->setup_vectors();

  this->auxiliary_scalar->clear_boundary_conditions();
  this->auxiliary_scalar->setup_boundary_conditions();

  /* Set boundary conditions */

  this->auxiliary_scalar->close_boundary_conditions();
  this->auxiliary_scalar->apply_boundary_conditions();

  this->auxiliary_scalar->set_solution_vectors_to_zero();

  this->flag_setup_auxiliary_scalar = false;
}




template <int dim>
void MagneticInduction<dim>::setup_matrices()
{
  if (parameters.verbose)
    *this->pcout << "  Magnetic Induction: Setting up the matrices...";

  TimerOutput::Scope  t(*this->computing_timer, "Magnetic Induction: Set-up matrices");

  ProjectionSolverBase<dim>::setup_matrices();

  if (parameters.verbose)
    *this->pcout << " done!" << std::endl << std::endl;
}



template <int dim>
void MagneticInduction<dim>::setup_vectors()
{
  if (parameters.verbose)
    *this->pcout << "  Magnetic Induction: Setting up the vectors...";

  TimerOutput::Scope  t(*this->computing_timer, "Magnetic Induction: Set-up vectors");

  ProjectionSolverBase<dim>::setup_vectors();

  if (parameters.verbose)
    *this->pcout << " done!" << std::endl << std::endl;
}



template <int dim>
void MagneticInduction<dim>::clear()
{
  pseudo_pressure->clear();

  ProjectionSolverBase<dim>::clear();
}



} // namespace Solvers



} // namespace RMHD

// Explicit instantiations
template void RMHD::Solvers::MagneticInduction<2>::setup();
template void RMHD::Solvers::MagneticInduction<3>::setup();

template void RMHD::Solvers::MagneticInduction<2>::setup_pseudo_pressure();
template void RMHD::Solvers::MagneticInduction<3>::setup_pseudo_pressure();

template void RMHD::Solvers::MagneticInduction<2>::setup_auxiliary_scalar();
template void RMHD::Solvers::MagneticInduction<3>::setup_auxiliary_scalar();

template void RMHD::Solvers::MagneticInduction<2>::setup_matrices();
template void RMHD::Solvers::MagneticInduction<3>::setup_matrices();

template void RMHD::Solvers::MagneticInduction<2>::setup_vectors();
template void RMHD::Solvers::MagneticInduction<3>::setup_vectors();

template void RMHD::Solvers::MagneticInduction<2>::clear();
template void RMHD::Solvers::MagneticInduction<3>::clear();
