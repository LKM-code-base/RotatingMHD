#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <rotatingMHD/global.h>
#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/vector_tools.h>

#include <array>

// Test serialization in parallel

using namespace dealii;

using namespace RMHD;

template<int dim>
double serialize_scalar_field()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  using VectorType = RMHD::LinearAlgebra::MPI::Vector;

  Entities::FE_ScalarField<dim, VectorType>
  field(1, tria, "Scalar Field");

  Functions::SquareFunction<dim> function;

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(3);

  field.setup_dofs();
  field.setup_vectors();

  field.setup_boundary_conditions();
  field.close_boundary_conditions(false);
  field.apply_boundary_conditions(false);

  RMHD::VectorTools::project(field,
                             function,
                             field.solution);

  SolutionTransferContainer<dim>  container;
  container.add_entity(field, false);
  container.serialize("scalar_field.mesh");

  VectorType  distributed_solution(field.distributed_vector);
  distributed_solution = field.solution;

  return(distributed_solution.l2_norm());

}



template<int dim>
double deserialize_scalar_field()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  using VectorType = RMHD::LinearAlgebra::MPI::Vector;

  Entities::FE_ScalarField<dim, VectorType>
  field(1, tria, "Scalar Field");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  SolutionTransferContainer<dim>  container;
  container.add_entity(field, false);
  container.deserialize(tria, "scalar_field.mesh");

  VectorType  distributed_solution(field.distributed_vector);
  distributed_solution = field.solution;

  return (distributed_solution.l2_norm());

}



template<int dim>
double serialize_vector_field()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  using VectorType = RMHD::LinearAlgebra::MPI::Vector;

  Entities::FE_VectorField<dim, VectorType>
  field(2, tria, "Vector Field");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(3);

  field.setup_dofs();
  field.setup_vectors();

  field.setup_boundary_conditions();
  field.close_boundary_conditions(false);
  field.apply_boundary_conditions(false);

  Functions::CosineFunction<dim> function(dim);

  RMHD::VectorTools::project(field,
                             function,
                             field.solution);

  SolutionTransferContainer<dim>  container;
  container.add_entity(field, false);
  container.serialize("vector_field.mesh");

  VectorType  distributed_solution(field.distributed_vector);
  distributed_solution = field.solution;

  return (distributed_solution.l2_norm());

}



template<int dim>
double deserialize_vector_field()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  using VectorType = RMHD::LinearAlgebra::MPI::Vector;

  Entities::FE_VectorField<dim, VectorType>
  field(2, tria, "Vector Field");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  SolutionTransferContainer<dim>  container;
  container.add_entity(field, false);
  container.deserialize(tria, "vector_field.mesh");

  VectorType  distributed_solution(field.distributed_vector);
  distributed_solution = field.solution;

  return (distributed_solution.l2_norm());
}



template<int dim>
std::array<double, 2> serialize_field_collection()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  using VectorType = RMHD::LinearAlgebra::MPI::Vector;

  Entities::FE_ScalarField<dim, VectorType>
  scalar_field(1, tria, "Scalar Field");

  Entities::FE_VectorField<dim, VectorType>
  vector_field(2, tria, "Vector Field");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(3);

  scalar_field.setup_dofs();
  scalar_field.setup_vectors();

  scalar_field.setup_boundary_conditions();
  scalar_field.close_boundary_conditions(false);
  scalar_field.apply_boundary_conditions(false);

  Functions::SquareFunction<dim> scalar_function;
  RMHD::VectorTools::project(scalar_field,
                             scalar_function,
                             scalar_field.solution);

  vector_field.setup_dofs();
  vector_field.setup_vectors();

  vector_field.setup_boundary_conditions();
  vector_field.close_boundary_conditions(false);
  vector_field.apply_boundary_conditions(false);

  Functions::CosineFunction<dim> vector_function(dim);
  RMHD::VectorTools::project(vector_field,
                             vector_function,
                             vector_field.solution);


  SolutionTransferContainer<dim>  container;
  container.add_entity(scalar_field, false);
  container.add_entity(vector_field, false);
  container.serialize("field_collection.mesh");

  std::array<double, 2> norms;
  VectorType  distributed_solution(scalar_field.distributed_vector);
  distributed_solution = scalar_field.solution;
  norms[0] = distributed_solution.l2_norm();

  distributed_solution.reinit(vector_field.distributed_vector);
  distributed_solution = vector_field.solution;
  norms[1] = distributed_solution.l2_norm();

  return (norms);

}



template<int dim>
std::array<double, 2> deserialize_field_collection()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  using VectorType = RMHD::LinearAlgebra::MPI::Vector;

  Entities::FE_ScalarField<dim, VectorType>
  scalar_field(1, tria, "Scalar Field");

  Entities::FE_VectorField<dim, VectorType>
  vector_field(2, tria, "Vector Field");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  SolutionTransferContainer<dim>  container;
  container.add_entity(scalar_field, false);
  container.add_entity(vector_field, false);
  container.deserialize(tria, "field_collection.mesh");

  std::array<double, 2> norms;
  VectorType  distributed_solution(scalar_field.distributed_vector);
  distributed_solution = scalar_field.solution;
  norms[0] = distributed_solution.l2_norm();

  distributed_solution.reinit(vector_field.distributed_vector);
  distributed_solution = vector_field.solution;
  norms[1] = distributed_solution.l2_norm();

  return (norms);

}



int main(int argc, char *argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize  mpi_initialization(argc, argv, 1);
    ConditionalOStream  pcout(std::cout,
                              Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    {
      double norm[2];
      norm[0] = serialize_scalar_field<2>();
      norm[1] = deserialize_scalar_field<2>();
      pcout << std::scientific << std::fabs(norm[0] - norm[1]) << std::endl;

      norm[0] = serialize_scalar_field<3>();
      norm[1] = deserialize_scalar_field<3>();
      pcout << std::scientific << std::fabs(norm[0] - norm[1])  << std::endl;

      norm[0] = serialize_vector_field<2>();
      norm[1] = deserialize_vector_field<2>();
      pcout << std::scientific << std::fabs(norm[0] - norm[1]) << std::endl;

      norm[0] = serialize_vector_field<3>();
      norm[1] = deserialize_vector_field<3>();
      pcout << std::scientific << std::fabs(norm[0] - norm[1]) << std::endl;

      norm[0] = serialize_vector_field<2>();
      norm[1] = deserialize_vector_field<2>();
      pcout << std::scientific << std::fabs(norm[0] - norm[1]) << std::endl;
    }
    {
      std::array<double, 2> norm[2];
      norm[0] = serialize_field_collection<3>();
      norm[1] = deserialize_field_collection<3>();
      pcout << std::scientific << std::fabs(norm[0][0] - norm[1][0]) << std::endl;
      pcout << std::scientific << std::fabs(norm[0][1] - norm[1][1]) << std::endl;
    }
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
