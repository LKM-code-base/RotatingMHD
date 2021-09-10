#include <rotatingMHD/solver_class.h>



namespace RMHD
{



namespace Solvers
{



using namespace dealii;



template<int dim>
SolverBase<dim>::SolverBase(
  TimeDiscretization::VSIMEXMethod          &time_stepping,
  const std::shared_ptr<Mapping<dim>>       external_mapping,
  const std::shared_ptr<ConditionalOStream> external_pcout,
  const std::shared_ptr<TimerOutput>        external_timer)
:
mpi_communicator(MPI_COMM_WORLD),
time_stepping(time_stepping),
flag_matrices_were_updated(true)
{
  // Initiating the internal Mapping instance.
  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping = std::make_shared<MappingQ<dim>>(1);

  // Initiating the internal ConditionalOStream instance.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout = std::make_shared<ConditionalOStream>(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

  // Initiating the internal TimerOutput instance.
  if (external_timer.get() != nullptr)
    computing_timer = external_timer;
  else
    computing_timer = std::make_shared<TimerOutput>(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times);
}



template<int dim>
ProjectionSolverBase<dim>::ProjectionSolverBase(
  TimeDiscretization::VSIMEXMethod                &time_stepping,
  const std::shared_ptr<Mapping<dim>>             external_mapping,
  const std::shared_ptr<ConditionalOStream>       external_pcout,
  const std::shared_ptr<TimerOutput>              external_timer)
:
SolverBase<dim>(time_stepping,
                external_mapping,
                external_pcout,
                external_timer),
norm_diffusion_step_rhs(std::numeric_limits<double>::lowest()),
norm_projection_step_rhs(std::numeric_limits<double>::lowest()),
flag_setup_auxiliary_scalar(true),
flag_mean_value_constrain(false)
{
  // Explicitly set the supply term's pointer to null
  ptr_supply_term = nullptr;
}



template <int dim>
void ProjectionSolverBase<dim>::setup()
{
  if (flag_setup_auxiliary_scalar)
    setup_auxiliary_scalar();

  this->setup_matrices();

  this->setup_vectors();

  assemble_constant_matrices();

  if (auxiliary_scalar->get_dirichlet_boundary_conditions().empty())
    flag_mean_value_constrain = true;

  this->flag_matrices_were_updated = true;

  if (this->time_stepping.get_step_number() == 0)
    zeroth_step();
}



template <int dim>
void ProjectionSolverBase<dim>::clear()
{
  // Pointers
  ptr_supply_term = nullptr;

  // Auxiliary scalar
  auxiliary_scalar->clear();

  // Preconditioners
  diffusion_step_preconditioner.reset();
  projection_step_preconditioner.reset();
  zeroth_step_preconditioner.reset();

  // Matrices
  diffusion_step_system_matrix.clear();
  diffusion_step_mass_matrix.clear();
  diffusion_step_stiffness_matrix.clear();
  diffusion_step_mass_plus_stiffness_matrix.clear();
  diffusion_step_advection_matrix.clear();
  projection_step_system_matrix.clear();
  zeroth_step_system_matrix.clear();

  // Vectors
  diffusion_step_rhs.clear();
  projection_step_rhs.clear();
  zeroth_step_rhs.clear();

  // Norms
  norm_projection_step_rhs  = std::numeric_limits<double>::lowest();
  norm_diffusion_step_rhs   = std::numeric_limits<double>::lowest();

  // Flags
  flag_setup_auxiliary_scalar       = true;
  flag_mean_value_constrain         = false;
  this->flag_matrices_were_updated  = true;
}


template <int dim>
void ProjectionSolverBase<dim>::set_supply_term(
  TensorFunction<1, dim> &supply_term)
{
  ptr_supply_term = &supply_term;
}



template <int dim>
void ProjectionSolverBase<dim>::zeroth_step()
{
  if (ptr_supply_term != nullptr)
    ptr_supply_term->set_time(this->time_stepping.get_start_time());

  assemble_zeroth_step();

  solve_zeroth_step();
}



template <int dim>
void ProjectionSolverBase<dim>::diffusion_step(const bool reinit_preconditioner)
{
  assemble_diffusion_step();

  solve_diffusion_step(reinit_preconditioner);
}



template <int dim>
void ProjectionSolverBase<dim>::projection_step(const bool reinit_preconditioner)
{
  assemble_projection_step();

  solve_projection_step(reinit_preconditioner);
}



template <int dim>
void ProjectionSolverBase<dim>::assemble_constant_matrices()
{
  assemble_constant_matrices_vector_field();

  assemble_constant_matrices_scalar_fields();
}



} // namespace Solvers



} // namespace RMHD



template struct RMHD::Solvers::SolverBase<2>;
template struct RMHD::Solvers::SolverBase<3>;

template struct RMHD::Solvers::ProjectionSolverBase<2>;
template struct RMHD::Solvers::ProjectionSolverBase<3>;