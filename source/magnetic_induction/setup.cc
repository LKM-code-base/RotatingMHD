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

  // Check for unsupported boundary conditions
  {
    std::vector<types::boundary_id> constrained_boundary_ids =
      magnetic_field->get_boundary_conditions().get_constrained_boundary_ids();

    const typename Entities::BoundaryConditionsBase<dim>::BCMapping
      tangential_component_boundary_conditions =
        magnetic_field->get_tangential_component_boundary_condition();

    std::vector<types::boundary_id> tangential_component_boundary_ids;

    for (auto const &tangential_component_boundary_condition :
          tangential_component_boundary_conditions)
      tangential_component_boundary_ids.push_back(
        tangential_component_boundary_condition.first);

    std::sort(constrained_boundary_ids.begin(),
              constrained_boundary_ids.end());

    std::sort(tangential_component_boundary_ids.begin(),
              tangential_component_boundary_ids.end());

    AssertThrow(tangential_component_boundary_ids
                  == constrained_boundary_ids,
                ExcMessage("Only boundary conditions imposed on the"
                           "tangential component are currently supported."));
  }

  // Homogeneous tangential boundary conditions are set at unconstrained
  // boundaries
  std::vector<types::boundary_id> unconstrained_boundary_ids =
    magnetic_field->get_boundary_conditions().get_unconstrained_boundary_ids();

  if (unconstrained_boundary_ids.size() != 0)
  {
    *this->pcout << std::endl
                  << "Warning: No boundary conditions for the magnetic field were "
                  << "assigned on the boundaries {";

    for (const auto &unconstrained_boundary_id : unconstrained_boundary_ids)
    {
      *this->pcout << unconstrained_boundary_id << ", ";
      magnetic_field->set_tangential_component_boundary_condition(
        unconstrained_boundary_id);
    }

    *this->pcout << "\b\b}. Homogeneous tangential component boundary "
                  "conditions will be assumed in order to properly "
                  "constraint the problem.\n"
                  << std::endl;
  }

  // Tangential component boundary conditions in the magnetic field
  // space translate into homogeneous Dirichlet boundary conditions
  // in the pseudo-pressure space
  for (const auto &tangential_component_bc :
        magnetic_field->get_tangential_component_boundary_condition())
    pseudo_pressure->set_dirichlet_boundary_condition(
                      tangential_component_bc.first);

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

  // Dirichlet boundary conditions in the pseudo pressure space
  // translate into homogeneous Dirichlet boundary conditions in the
  // auxiliary scalar space
  for (const auto &dirichlet_bc : pseudo_pressure->get_dirichlet_boundary_conditions())
    this->auxiliary_scalar->set_dirichlet_boundary_condition(dirichlet_bc.first);

  // The remaining unconstrained boundaries in the auxiliary scalar
  // space are set to homogeneous Neumann boundary conditions
  std::vector<types::boundary_id> unconstrained_boundary_ids =
    pseudo_pressure->get_boundary_conditions().get_unconstrained_boundary_ids();

  for (const auto &unconstrained_boundary_id: unconstrained_boundary_ids)
    this->auxiliary_scalar->set_neumann_boundary_condition(unconstrained_boundary_id);


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
