#include <rotatingMHD/global.h>
#include <rotatingMHD/problem_class.h>

#include <deal.II/base/quadrature_lib.h>

#include <exception>
#include <filesystem>
#include <string>

namespace RMHD
{

using namespace dealii;

template<int dim>
SolutionTransferContainer<dim>::SolutionTransferContainer()
:
error_vector_size(0)
{}

template<int dim>
void SolutionTransferContainer<dim>::add_entity(
std::shared_ptr<Entities::EntityBase<dim>> entity, bool flag)
{
  entities.emplace_back(std::make_pair(entity.get(), flag));
  if (flag)
    error_vector_size += 1;
}

/*
 *
 * @attention Delete this once the problem solver is working
 *
 */
template<int dim>
Problem<dim>::Problem(const RunTimeParameters::ProblemBaseParameters &prm)
:
mpi_communicator(MPI_COMM_WORLD),
prm(prm),
triangulation(mpi_communicator,
              typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::smoothing_on_refinement |
              Triangulation<dim>::smoothing_on_coarsening)),
mapping(std::make_shared<MappingQ<dim>>(prm.mapping_degree,
                                        prm.mapping_interior_cells)),
pcout(std::make_shared<ConditionalOStream>(std::cout,
      (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))),
computing_timer(
  std::make_shared<TimerOutput>(mpi_communicator,
                                *pcout,
                                TimerOutput::summary,
                                TimerOutput::wall_times))
{
  if (!std::filesystem::exists(prm.graphical_output_directory) &&
      Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
  {
    try
    {
      std::filesystem::create_directories(prm.graphical_output_directory);
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception in the creation of the output directory: "
                << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
    catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                  << std::endl;
      std::cerr << "Unknown exception in the creation of the output directory!"
                << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
  }
}



template <int dim>
void Problem<dim>::clear()
{
  container.clear();

  triangulation.clear();
}



template <int dim>
void Problem<dim>::project_function
(const Function<dim>                             &function,
 const std::shared_ptr<Entities::EntityBase<dim>> entity,
 LinearAlgebra::MPI::Vector                      &vector)
{
  Assert(function.n_components == entity->n_components,
         ExcMessage("The number of components of the function does not those "
                    "of the entity"));
  #ifdef USE_PETSC_LA
    LinearAlgebra::MPI::Vector
    tmp_vector(entity->locally_owned_dofs, mpi_communicator);
  #else
    LinearAlgebra::MPI::Vector
    tmp_vector(entity->locally_owned_dofs);
  #endif

  VectorTools::project(*(this->mapping),
                      *(entity->dof_handler),
                       entity->constraints,
                       QGauss<dim>(entity->fe_degree + 2),
                       function,
                       tmp_vector);

  vector = tmp_vector;
}



template <int dim>
void Problem<dim>::interpolate_function
(const Function<dim>                             &function,
 const std::shared_ptr<Entities::EntityBase<dim>> entity,
 LinearAlgebra::MPI::Vector                      &vector)
{
  Assert(function.n_components == entity->n_components,
         ExcMessage("The number of components of the function does not those "
                    "of the entity"));
  #ifdef USE_PETSC_LA
    LinearAlgebra::MPI::Vector
    tmp_vector(entity->locally_owned_dofs, mpi_communicator);
  #else
    LinearAlgebra::MPI::Vector
    tmp_vector(entity->locally_owned_dofs);
  #endif

  VectorTools::interpolate(*entity->dof_handler,
                           function,
                           tmp_vector);

  vector = tmp_vector;
}



template <int dim>
void Problem<dim>::set_initial_conditions
(std::shared_ptr<Entities::EntityBase<dim>> entity,
 Function<dim>                              &function,
 const TimeDiscretization::VSIMEXMethod     &time_stepping,
 const bool                                 boolean)
{
  #ifdef USE_PETSC_LA
    LinearAlgebra::MPI::Vector
    tmp_old_solution(entity->locally_owned_dofs, mpi_communicator);
  #else
    LinearAlgebra::MPI::Vector
    tmp_old_solution(entity->locally_owned_dofs);
  #endif

  function.set_time(time_stepping.get_start_time());

  if (!boolean)
  {
    VectorTools::project(*entity->dof_handler,
                          entity->constraints,
                          QGauss<dim>(entity->fe_degree + 2),
                          function,
                          tmp_old_solution);

    entity->old_solution = tmp_old_solution;
  }
  else
  {
    LinearAlgebra::MPI::Vector tmp_old_old_solution(tmp_old_solution);

    VectorTools::project(*entity->dof_handler,
                          entity->constraints,
                          QGauss<dim>(entity->fe_degree + 2),
                          function,
                          tmp_old_old_solution);

    function.advance_time(time_stepping.get_next_step_size());

    VectorTools::project(*entity->dof_handler,
                          entity->constraints,
                          QGauss<dim>(entity->fe_degree + 2),
                          function,
                          tmp_old_solution);

    entity->old_old_solution = tmp_old_old_solution;
    entity->old_solution     = tmp_old_solution;
  }
}

template <int dim>
void Problem<dim>::compute_error(
  LinearAlgebra::MPI::Vector                  &error_vector,
  std::shared_ptr<Entities::EntityBase<dim>>  entity,
  Function<dim>                               &exact_solution)
{
  AssertThrow(error_vector.local_size() ==
                entity->solution.local_size(),
              ExcMessage("The vectors do not match in size"));

  LinearAlgebra::MPI::Vector  distributed_solution_vector;
  LinearAlgebra::MPI::Vector  distributed_error_vector;

  distributed_solution_vector.reinit(entity->distributed_vector);
  distributed_error_vector.reinit(entity->distributed_vector);

  distributed_solution_vector = entity->solution;
  VectorTools::interpolate(*mapping,
                           *entity->dof_handler,
                           exact_solution,
                           distributed_error_vector);
  entity->hanging_nodes.distribute(distributed_error_vector);

  distributed_error_vector.add(-1.0, distributed_solution_vector);

  for (unsigned int i = distributed_error_vector.local_range().first;
       i < distributed_error_vector.local_range().second; ++i)
    if (distributed_error_vector(i) < 0)
      distributed_error_vector(i) *= -1.0;

  error_vector = distributed_error_vector;
}

template <int dim>
double Problem<dim>::compute_next_time_step
(const TimeDiscretization::VSIMEXMethod &time_stepping,
 const double                           cfl_number,
 const double                           max_cfl_number) const
{
  if (!prm.time_discretization_parameters.adaptive_time_stepping ||
      time_stepping.get_step_number() == 0)
    return time_stepping.get_next_step_size();

  return max_cfl_number / cfl_number *
         time_stepping.get_next_step_size();
}

template <int dim>
void Problem<dim>::adaptive_mesh_refinement()
{
  Assert(!container.empty(),
         ExcMessage("The entities container is empty."))

  using SolutionTransferType =
  parallel::distributed::SolutionTransfer<dim, LinearAlgebra::MPI::Vector>;

  using TransferVectorType =
  std::vector<const LinearAlgebra::MPI::Vector *>;

  std::vector<SolutionTransferType> solution_transfers;

  /*! Initiates the objects responsible for the solution transfer */
  for (auto const &entity: container.entities)
    solution_transfers.emplace_back(*(entity.first->dof_handler));

  {
    TimerOutput::Scope t(*computing_timer,
                         "Problem: Adaptive mesh refinement Pt. 1");

    *pcout << std::endl
           << " Preparing coarsening and refining..." << std::endl;

    // Initiates the estimated error per cell of each entity, which
    // is to be considered
    std::vector<Vector<float>> estimated_errors_per_cell(
      container.get_error_vector_size(),
      Vector<float>(triangulation.n_active_cells()));

    // Initiates the estimated error per cell used in the refinement.
    // It is composed by the equally weighted sum of the estimated
    // error per cell of each entity to be considered
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    unsigned int j = 0;

    // Computes the estimated error per cell of all the pertinent
    // entities
    for (unsigned int i = 0; i < container.entities.size(); ++i)
    {
      if (container.entities[i].second)
      {
        KellyErrorEstimator<dim>::estimate(
          *(container.entities[i].first->dof_handler),
          QGauss<dim-1>(container.entities[i].first->fe_degree + 1),
          std::map<types::boundary_id, const Function<dim> *>(),
          container.entities[i].first->solution,
          estimated_errors_per_cell[j],
          ComponentMask(),
          nullptr,
          0,
          triangulation.locally_owned_subdomain());

        j += 1;
      }
      else
        continue;
    }

    // Reset the estimated error per cell and fills it with the
    // equally weighted sum of the estimated error per cell of each
    // entity to be considered
    estimated_error_per_cell = 0.;

    for (auto const &error_vector: estimated_errors_per_cell)
      estimated_error_per_cell.add(1.0 / container.get_error_vector_size(),
                                   error_vector);

    // Indicates which cells are to be refine/coarsen
    parallel::distributed::
    GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation,
      estimated_error_per_cell,
      prm.spatial_discretization_parameters.cell_fraction_to_coarsen,
      prm.spatial_discretization_parameters.cell_fraction_to_refine);

    // Clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_global_levels() > prm.spatial_discretization_parameters.n_maximum_levels)
      for (auto cell: triangulation.active_cell_iterators_on_level(
                        prm.spatial_discretization_parameters.n_maximum_levels))
        cell->clear_refine_flag();

    // Clear coarsen flags if level decreases minimum
    for (auto cell: triangulation.active_cell_iterators_on_level(
                      prm.spatial_discretization_parameters.n_minimum_levels))
        cell->clear_coarsen_flag();

    // Count number of cells to be refined and coarsened
    unsigned int local_cell_counts[2] = {0, 0};
    for (auto cell: triangulation.active_cell_iterators())
        if (cell->is_locally_owned() && cell->refine_flag_set())
            local_cell_counts[0] += 1;
        else if (cell->is_locally_owned() && cell->coarsen_flag_set())
            local_cell_counts[1] += 1;

    unsigned int global_cell_counts[2];
    Utilities::MPI::sum(local_cell_counts, mpi_communicator, global_cell_counts);

    *pcout << "   Number of cells set for refinement: " << global_cell_counts[0] << std::endl
           << "   Number of cells set for coarsening: " << global_cell_counts[1] << std::endl;

    // Stores the current solutions into std::vector declared below
    // and prepare each entry for the solution transfer
    std::vector<TransferVectorType> x_solutions;

    for (auto const &entity : container.entities)
    {
      TransferVectorType x_solution(3);
      x_solution[0] = &(entity.first)->solution;
      x_solution[1] = &(entity.first)->old_solution;
      x_solution[2] = &(entity.first)->old_old_solution;
      x_solutions.emplace_back(x_solution);
    }

    triangulation.prepare_coarsening_and_refinement();
    for (unsigned int i = 0; i < solution_transfers.size(); ++i)
      solution_transfers[i].prepare_for_coarsening_and_refinement(
        x_solutions[i]);

    // Execute the mesh refinement/coarsening
    *pcout << " Executing coarsening and refining..." << std::endl;
    triangulation.execute_coarsening_and_refinement();
  }

  *pcout << "   Number of global active cells:      "
         << triangulation.n_global_active_cells() << std::endl;

  std::vector<unsigned int> locally_active_cells(triangulation.n_global_levels());
  for (unsigned int level = 0; level < triangulation.n_levels(); ++level)
    for (auto cell: triangulation.active_cell_iterators_on_level(level))
      if (cell->is_locally_owned())
        locally_active_cells[level] += 1;
  *pcout << "   Number of cells on each (level):    ";
  for (unsigned int level=0; level < triangulation.n_global_levels(); ++level)
  {
      *pcout << Utilities::MPI::sum(locally_active_cells[level], mpi_communicator)
             << " (" << level << ")" << ", ";
  }
  *pcout << "\b\b \n" << std::endl;

  int n_total_dofs = 0;

  // Reinitiate the entities to accomodate to the new mesh
  for (auto &entity: container.entities)
  {
    (entity.first)->setup_dofs();

    if (!entity.first->is_child_entity())
    {
      *pcout << std::setw(60)
             << (" Number of degrees of freedom of the \""
                + (entity.first)->name + "\" entity")
             << " = "
             << (entity.first)->dof_handler->n_dofs()
             << std::endl;

      n_total_dofs += (entity.first)->dof_handler->n_dofs();
    }

    (entity.first)->apply_boundary_conditions();
    (entity.first)->reinit();
  }

  *pcout << std::setw(60)
         << " Number of total degrees of freedom"
         << " = "
         << n_total_dofs << std::endl << std::endl;

  {
    TimerOutput::Scope t(*computing_timer,
                         "Problem: Adaptive mesh refinement Pt. 2");

    for (unsigned int i = 0; i < container.entities.size(); ++i)
    {
      // Temporary vectors to extract the interpolated solutions back
      // into the entities
      LinearAlgebra::MPI::Vector  distributed_tmp_solution;
      LinearAlgebra::MPI::Vector  distributed_tmp_old_solution;
      LinearAlgebra::MPI::Vector  distributed_tmp_old_old_solution;

      #ifdef USE_PETSC_LA
        distributed_tmp_solution.reinit(
          (container.entities[i].first)->locally_owned_dofs,
          (container.entities[i].first)->mpi_communicator);
      #else
        distributed_tmp_solution.reinit(
          (container.entities[i].first)->locally_owned_dofs,
          (container.entities[i].first)->locally_relevant_dofs,
          (container.entities[i].first)->mpi_communicator,
          true);
      #endif

      distributed_tmp_old_solution.reinit(distributed_tmp_solution);
      distributed_tmp_old_old_solution.reinit(distributed_tmp_solution);

      std::vector<LinearAlgebra::MPI::Vector *>  tmp(3);
      tmp[0] = &(distributed_tmp_solution);
      tmp[1] = &(distributed_tmp_old_solution);
      tmp[2] = &(distributed_tmp_old_old_solution);

      // Interpolates and apply constraines to the temporary vectors
      solution_transfers[i].interpolate(tmp);

      (container.entities[i].first)->constraints.distribute(distributed_tmp_solution);
      (container.entities[i].first)->constraints.distribute(distributed_tmp_old_solution);
      (container.entities[i].first)->constraints.distribute(distributed_tmp_old_old_solution);

      // Passes the interpolated vectors to the entities' vector instances
      (container.entities[i].first)->solution          = distributed_tmp_solution;
      (container.entities[i].first)->old_solution      = distributed_tmp_old_solution;
      (container.entities[i].first)->old_old_solution  = distributed_tmp_old_old_solution;
    }
  }
}


} // namespace RMHD

template struct RMHD::SolutionTransferContainer<2>;
template struct RMHD::SolutionTransferContainer<3>;

template class RMHD::Problem<2>;
template class RMHD::Problem<3>;
