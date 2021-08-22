#include <rotatingMHD/solver_class.h>



namespace RMHD
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
time_stepping(time_stepping)
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
  TimeDiscretization::VSIMEXMethod          &time_stepping,
  const std::shared_ptr<Mapping<dim>>       external_mapping,
  const std::shared_ptr<ConditionalOStream> external_pcout,
  const std::shared_ptr<TimerOutput>        external_timer)
:
SolverBase<dim>(time_stepping,
                external_mapping,
                external_pcout,
                external_timer)
{}

} // namespace RMHD



template struct RMHD::SolverBase<2>;
template struct RMHD::SolverBase<3>;

template struct RMHD::ProjectionSolverBase<2>;
template struct RMHD::ProjectionSolverBase<3>;