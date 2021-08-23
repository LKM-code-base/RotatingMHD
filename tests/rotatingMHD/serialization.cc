#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <rotatingMHD/global.h>
#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/vector_tools.h>

namespace RMHD
{

template<int dim>
void serialize_scalar_field()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  Entities::FE_ScalarField<dim, LinearAlgebra::MPI::Vector>
  field(1, tria, "Scalar Field");

  Functions::SquareFunction<dim> function;

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(3);

  field.setup_dofs();
  field.setup_vectors();

  field.setup_boundary_conditions();
  field.close_boundary_conditions(false);
  field.apply_boundary_conditions(false);

  VectorTools::project(field,
                       function,
                       field.solution);
  VectorTools::interpolate(field,
                           function,
                           field.old_solution);

  SolutionTransferContainer<dim>  container;
  container.add_entity(field, false);
  container.serialize("scalar_field.mesh");
}



template<int dim>
void deserialize_scalar_field()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  Entities::FE_ScalarField<dim, LinearAlgebra::MPI::Vector>
  field(1, tria, "Scalar Field");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  SolutionTransferContainer<dim>  container;
  container.add_entity(field, false);
  container.deserialize(tria, "scalar_field.mesh");

  Point<dim> point;
  for (unsigned d=0; d<dim; ++d)
    point[d] = 0.5;

  Functions::SquareFunction<dim> function;
  const double function_value{function.value(point)};

  std::cout << std::fabs(field.point_value(point) - function_value) << std::endl;
}


}  // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize  mpi_initialization(argc, argv);
    dealii::deallog.depth_console(0);

    RMHD::serialize_scalar_field<2>();
    RMHD::deserialize_scalar_field<2>();

    RMHD::serialize_scalar_field<3>();
    RMHD::deserialize_scalar_field<3>();
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
