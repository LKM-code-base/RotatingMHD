#include <rotatingMHD/boundary_conditions.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_generator.h>

namespace RMHD
{

namespace Entities
{

template<int dim>
void scalar_boundary_condition_test()
{
  Triangulation<dim>            tria;
  ScalarBoundaryConditions<dim> scalar_bcs01(tria);
  scalar_bcs01.clear();

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  std::shared_ptr<Function<dim>>  ptr01 =
      std::make_shared<Functions::ExpFunction<dim>>();
  std::shared_ptr<Function<dim>>  ptr02 =
      std::make_shared<Functions::CutOffFunctionC1<dim>>();

  scalar_bcs01.extract_boundary_ids();
  std::cout << "unconstrained_ids: ";
  for (const auto &id: scalar_bcs01.get_unconstrained_boundary_ids())
    std::cout << id << ", ";
  std::cout << std::endl;

  scalar_bcs01.set_periodic_bc(0, 1, 0);
  scalar_bcs01.set_dirichlet_bc(2, ptr01, true);
  scalar_bcs01.set_neumann_bc(3, ptr02);
  scalar_bcs01.set_time(1.0);
  std::cout << "Time of ptr01 = " << ptr01->get_time() << std::endl;
  std::cout << "Time of ptr02 = " << ptr02->get_time() << std::endl;

  std::cout << "unconstrained_ids: ";
  for (const auto &id: scalar_bcs01.get_unconstrained_boundary_ids())
    std::cout << id << ", ";
  std::cout << std::endl;
  scalar_bcs01.print_summary(std::cout, "Scalar 01");

  ScalarBoundaryConditions<dim> scalar_bcs02(tria);
  scalar_bcs02.copy(scalar_bcs01);
  scalar_bcs02.print_summary(std::cout, "Scalar 02");

  ScalarBoundaryConditions<dim> scalar_bcs03(tria);
  scalar_bcs03.copy(scalar_bcs01);
  scalar_bcs03.clear();
  scalar_bcs03.extract_boundary_ids();
  scalar_bcs03.set_datum_at_boundary();
  scalar_bcs03.print_summary(std::cout, "Scalar 03");

  ScalarBoundaryConditions<dim> scalar_bcs04(tria);
  scalar_bcs04.clear();
  scalar_bcs04.extract_boundary_ids();
  scalar_bcs04.set_periodic_bc(0, 1, 0);
  scalar_bcs04.set_dirichlet_bc(2, ptr01, true);
  scalar_bcs04.set_neumann_bc(3, ptr02, true);
  scalar_bcs04.set_time(10.0);
  scalar_bcs04.print_summary(std::cout, "Scalar 03");
  std::cout << "Time of ptr01 = " << ptr01->get_time() << std::endl;
  std::cout << "Time of ptr02 = " << ptr02->get_time() << std::endl;

}



template<int dim>
void vector_boundary_condition_test()
{
  Triangulation<dim>            tria;
  VectorBoundaryConditions<dim> vector_bcs01(tria);
  vector_bcs01.clear();

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  std::shared_ptr<Function<dim>>  ptr01 =
      std::make_shared<Functions::CutOffFunctionC1<dim>>(1.0, Point<dim>(), dim);
  std::shared_ptr<TensorFunction<1, dim>>  ptr02 =
      std::make_shared<ZeroTensorFunction<1, dim>>();
  std::shared_ptr<Function<dim>>  ptr03 =
      std::make_shared<Functions::CosineFunction<dim>>(dim);
  std::shared_ptr<Function<dim>>  ptr04 =
      std::make_shared<Functions::ConstantFunction<dim>>(1.0, dim);

  vector_bcs01.extract_boundary_ids();
  std::cout << "unconstrained_ids: ";
  for (const auto &id: vector_bcs01.get_unconstrained_boundary_ids())
    std::cout << id << ", ";
  std::cout << std::endl;

  vector_bcs01.set_periodic_bc(0, 1, 0);
  vector_bcs01.set_dirichlet_bc(2, ptr01, true);
  vector_bcs01.set_neumann_bc(3, ptr02);
  vector_bcs01.set_time(1.0);
  std::cout << "Time of ptr01 = " << ptr01->get_time() << std::endl;
  std::cout << "Time of ptr02 = " << ptr02->get_time() << std::endl;

  std::cout << "unconstrained_ids: ";
  for (const auto &id: vector_bcs01.get_unconstrained_boundary_ids())
    std::cout << id << ", ";
  std::cout << std::endl;
  vector_bcs01.print_summary(std::cout, "Vector 01");

  VectorBoundaryConditions<dim> vector_bcs02(tria);
  vector_bcs02.copy(vector_bcs01);
  vector_bcs02.print_summary(std::cout, "Vector 02");

  VectorBoundaryConditions<dim> vector_bcs03(tria);
  vector_bcs03.clear();
  vector_bcs03.extract_boundary_ids();

  std::cout << "unconstrained_ids: ";
  for (const auto &id: vector_bcs03.get_unconstrained_boundary_ids())
    std::cout << id << ", ";
  std::cout << std::endl;

  vector_bcs03.set_dirichlet_bc(0, ptr01, true);
  vector_bcs03.set_neumann_bc(1, ptr02, true);
  vector_bcs03.set_tangential_flux_bc(2, ptr03, true);
  if (dim==3)
    vector_bcs03.set_normal_flux_bc(4, ptr04, true);
  vector_bcs03.set_time(10.0);
  vector_bcs03.print_summary(std::cout, "Vector 03");
  std::cout << "Time of ptr01 = " << ptr01->get_time() << std::endl;
  std::cout << "Time of ptr02 = " << ptr02->get_time() << std::endl;
  std::cout << "Time of ptr03 = " << ptr03->get_time() << std::endl;
  if (dim==3)
    std::cout << "Time of ptr04 = " << ptr04->get_time() << std::endl;
}


}  // namespace Entities

}  // namespace RMHD

int main(void)
{
  try
  {
    dealii::deallog.depth_console(0);

    RMHD::Entities::scalar_boundary_condition_test<2>();
    RMHD::Entities::scalar_boundary_condition_test<3>();

    RMHD::Entities::vector_boundary_condition_test<2>();
    RMHD::Entities::vector_boundary_condition_test<3>();

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
