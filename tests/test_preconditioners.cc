#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
# define FORCE_USE_OF_TRILINOS
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
# include<deal.II/lac/petsc_precondition.h>
# define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
# include <deal.II/lac/trilinos_precondition.h>

#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif

enum class PreconditionerType
{
  /*!
   * @brief Incomplete LU decomposition preconditioning.
   */
  ILU,

  /*!
   * @brief Algebraic multigrid preconditioning.
   */
  AMG,

  /*!
   * @brief Jacobi preconditioning.
   */
  Jacobi,

  /*!
   * @brief Symmetric Sucessive overrelaxation preconditioning. The system
   * matrix must be symmetric to apply this preconditioner.
   */
  SSOR

};


} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace PreconditionerTest
{

using namespace dealii;


template <int dim>
class LaplaceProblem
{
public:
  LaplaceProblem
  (const typename LA::PreconditionerType &precondtioner_type,
   unsigned int fe_degree = 1,
   bool         apply_mean_value_constraint = false);

  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector       locally_relevant_solution;
  LA::MPI::Vector       system_rhs;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

  const LA::PreconditionerType  preconditioner_type;
  const bool  apply_mean_value_constraint;
};




template <int dim>
LaplaceProblem<dim>::LaplaceProblem
(const typename LA::PreconditionerType  &preconditioner_type,
 unsigned int fe_degree,
 bool         apply_mean_value_constraint)
:
mpi_communicator(MPI_COMM_WORLD),
triangulation(mpi_communicator,
              typename Triangulation<dim>::MeshSmoothing(
                  Triangulation<dim>::smoothing_on_refinement |
                  Triangulation<dim>::smoothing_on_coarsening)),
fe(fe_degree),
dof_handler(triangulation),
pcout(std::cout,
      (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
computing_timer(mpi_communicator,
                pcout,
                TimerOutput::never,
                TimerOutput::wall_times),
preconditioner_type(preconditioner_type),
apply_mean_value_constraint(apply_mean_value_constraint)
{}




template <int dim>
void LaplaceProblem<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  constraints.clear();

  constraints.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  if (apply_mean_value_constraint)
  {
    IndexSet    boundary_dofs;
    DoFTools::extract_boundary_dofs(dof_handler,
                                    ComponentMask(fe.n_components(), true),
                                    boundary_dofs);

    types::global_dof_index local_idx = numbers::invalid_dof_index;
    IndexSet::ElementIterator idx = boundary_dofs.begin();
    IndexSet::ElementIterator endidx = boundary_dofs.end();
    for(; idx != endidx; ++idx)
      if (constraints.can_store_line(*idx) && !constraints.is_constrained(*idx))
      {
        local_idx = *idx;
        break;
      }
    // Make a reduction to find the smallest index (processors that
    // found a larger candidate just happened to not be able to store
    // that index with the minimum value). Note that it is possible that
    // some processors might not be able to find a potential DoF, for
    // example because they don't own any DoFs. On those processors we
    // will use dof_handler.n_dofs() when building the minimum (larger
    // than any valid DoF index).
    const types::global_dof_index global_idx
    = Utilities::MPI::min(
        (local_idx != numbers::invalid_dof_index) ? local_idx : dof_handler.n_dofs(),
        mpi_communicator);

    Assert(global_idx < dof_handler.n_dofs(),
           ExcMessage("Error, couldn't find a DoF to constrain."));

    // Finally set this DoF to zero (if we care about it):
    if (constraints.can_store_line(global_idx))
    {
        Assert(!constraints.is_constrained(global_idx),
               ExcInternalError());
        constraints.add_line(global_idx);
    }
  }
  else
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
}




template <int dim>
void LaplaceProblem<dim>::assemble_system()
{
  TimerOutput::Scope t(computing_timer, "assembly");

  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      cell_matrix = 0.;
      cell_rhs    = 0.;

      fe_values.reinit(cell);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const double rhs_value =
          (fe_values.quadrature_point(q_point)[1] >
               0.5 +
               0.25 * std::sin(4.0 * numbers::PI *
                               fe_values.quadrature_point(q_point)[0]) ?
               1. : -1.);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                 fe_values.shape_grad(j, q_point) *
                                 fe_values.JxW(q_point);

          cell_rhs(i) += rhs_value *
                         fe_values.shape_value(i, q_point) *
                         fe_values.JxW(q_point);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                             cell_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}




template <int dim>
void LaplaceProblem<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "solve");
  LA::MPI::Vector    completely_distributed_solution(locally_owned_dofs,
                                                     mpi_communicator);

  SolverControl solver_control(1000, 1e-6);

  #ifdef USE_PETSC_LA
    LA::SolverCG solver(solver_control, mpi_communicator);
  #else
    LA::SolverCG solver(solver_control);
  #endif

  switch (preconditioner_type)
  {
    case LA::PreconditionerType::AMG:
    {
      LA::MPI::PreconditionAMG preconditioner;
      LA::MPI::PreconditionAMG::AdditionalData data;

      #ifdef USE_PETSC_LA
        data.symmetric_operator = true;
      #else
        /* Trilinos defaults are good */
      #endif
      preconditioner.initialize(system_matrix, data);

      solver.solve(system_matrix,
                   completely_distributed_solution,
                   system_rhs,
                   preconditioner);
      break;
    }
    case LA::PreconditionerType::ILU:
    {
      LA::MPI::PreconditionIC preconditioner;
      LA::MPI::PreconditionIC::AdditionalData data;

      #ifdef USE_PETSC_LA
        data.levels = 1;
      #else
        data.ic_fill = 1;
        data.overlap = 2;
        data.ic_atol = 1e-5;
        data.ic_rtol = 1.01;
      #endif
      preconditioner.initialize(system_matrix, data);

      solver.solve(system_matrix,
                   completely_distributed_solution,
                   system_rhs,
                   preconditioner);
      break;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Preconditioner type is not implemented."));
      break;
    }
  }

  pcout << "   Solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  if (apply_mean_value_constraint)
  {
    /*
     * step 1: set non-distributed solution to preliminary solution in
     *         order to apply the constraints and to compute mean value
     */
    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
    /*
     * step 2: compute mean value
     */
    const LA::MPI::Vector::value_type mean_value
      = VectorTools::compute_mean_value(dof_handler,
                                        QGauss<dim>(fe.degree + 1),
                                        locally_relevant_solution,
                                        0);
    /*
     * step 3: substract mean value from distributed solution
     */
    completely_distributed_solution.add(-mean_value);
    /*
     * step 4: assign non-distributed solution
     */
    locally_relevant_solution = completely_distributed_solution;
  }
  else
  {
    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }
}




template <int dim>
void LaplaceProblem<dim>::refine_grid()
{
  TimerOutput::Scope t(computing_timer, "refine");

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    QGauss<dim - 1>(fe.degree + 1),
    std::map<types::boundary_id, const Function<dim> *>(),
    locally_relevant_solution,
    estimated_error_per_cell);
  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
    triangulation, estimated_error_per_cell, 0.3, 0.03);
  triangulation.execute_coarsening_and_refinement();
}




template <int dim>
void LaplaceProblem<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "u");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "solution", cycle, mpi_communicator, 2, 8);
}




template <int dim>
void LaplaceProblem<dim>::run()
{
  pcout << "Running with "
        #ifdef USE_PETSC_LA
        << "PETSc"
        #else
        << "Trilinos"
        #endif
        << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << " MPI rank(s)..." << std::endl;

  if (apply_mean_value_constraint)
    pcout << "Using mean value constraint and ";
  else
    pcout << "Using Dirichlet boundary conditions and ";

  switch (preconditioner_type)
  {
    case LA::PreconditionerType::AMG:
      pcout << "AMG preconditioning..." << std::endl;
      break;
    case LA::PreconditionerType::ILU:
      pcout << "AMG preconditioning..." << std::endl;
      break;
    default:
      AssertThrow(false, ExcMessage("Preconditioner type not implemented."));
      break;
  }

  const unsigned int n_cycles = 3;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
  {
    pcout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);

      {
        std::string   filename = "nsbench2.inp";
        std::ifstream file(filename);
        Assert(file, ExcFileNotOpen(filename.c_str()));
        grid_in.read_ucd(file);
      }

      for (auto &cell: triangulation.active_cell_iterators())
        if (cell->at_boundary())
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
              cell->face(f)->set_boundary_id(0);

      triangulation.refine_global(5);
    }
    else
      refine_grid();

    setup_system();

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    assemble_system();
    solve();

    if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
    {
      TimerOutput::Scope t(computing_timer, "output");
      output_results(cycle);
    }

    computing_timer.reset();

    pcout << std::endl;
  }
}

} // namespace PreconditionerTest

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace PreconditionerTest;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    {
      int rank = 0;
      {
        const int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        AssertThrowMPI(ierr);
      }
      if (rank == 0)
      {
        namespace fs = std::filesystem;
        if(!fs::exists("nsbench2.inp"))
          fs::create_symlink("../applications/nsbench2.inp", "nsbench2.inp");
        else if (!fs::is_symlink("nsbench2.inp"))
        {
          const bool ierr = fs::remove("nsbench2.inp");
          AssertThrow(ierr, ExcInternalError())
          fs::create_symlink("../applications/nsbench2.inp", "nsbench2.inp");
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }

    {
      LaplaceProblem<2> laplace_problem_2d(LA::PreconditionerType::AMG);
      laplace_problem_2d.run();
    }
    {
      LaplaceProblem<2> laplace_problem_2d(LA::PreconditionerType::ILU);
      laplace_problem_2d.run();
    }
    {
      LaplaceProblem<2> laplace_problem_2d(LA::PreconditionerType::AMG,
                                           1,
                                           true);
      laplace_problem_2d.run();
    }
    {
      LaplaceProblem<2> laplace_problem_2d(LA::PreconditionerType::ILU,
                                           1,
                                           true);
      laplace_problem_2d.run();
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
