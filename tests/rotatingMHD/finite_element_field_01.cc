#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/vector_tools.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_generator.h>

// Test of functionality of classes FE_ScalarField and FE_VectorField in serial

using namespace dealii;
using namespace RMHD;

template<int dim>
void test_scalar_field()
{
  Triangulation<dim>  tria;

  Entities::FE_ScalarField<dim, Vector<double>> field_01(1, tria, "Scalar Field 01");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  // Test setup
  field_01.setup_dofs();
  field_01.setup_vectors();

  std::shared_ptr<Function<dim>>  ptr01 =
      std::make_shared<Functions::ExpFunction<dim>>();
  std::shared_ptr<Function<dim>>  ptr02 =
      std::make_shared<Functions::CutOffFunctionC1<dim>>();

  // Test assignment of boundary conditions
  field_01.setup_boundary_conditions();
  field_01.set_periodic_boundary_condition(0, 1, 0);
  field_01.set_dirichlet_boundary_condition(2, ptr01, true);
  field_01.set_neumann_boundary_condition(3, ptr02);
  field_01.close_boundary_conditions(true);
  field_01.apply_boundary_conditions(true);

  // Test copying and modification of boundary conditions
  Entities::FE_ScalarField<dim, Vector<double>> field_02(field_01, "Scalar Field 02");
  field_02.setup_dofs();
  field_02.setup_vectors();
  field_02.setup_boundary_conditions();
  Entities::BoundaryConditionsBase<dim> &bc_02 = field_02.get_boundary_conditions();
  const Entities::BoundaryConditionsBase<dim> &bc_01 = field_01.get_boundary_conditions();
  bc_02.copy(bc_01);
  for (auto &dirichlet_bc: bc_02.dirichlet_bcs)
    dirichlet_bc.second = std::make_shared<Functions::ZeroFunction<dim>>();
  const std::vector<types::boundary_id> unconstrained_boundary_ids =
      bc_02.get_unconstrained_boundary_ids();
  for (const auto id: unconstrained_boundary_ids)
    field_02.set_neumann_boundary_condition(id, ptr01, true);
  field_02.close_boundary_conditions(true);
  field_02.apply_boundary_conditions(true);

  // Test point_value
  tria.refine_global(3);
  Entities::FE_ScalarField<dim, Vector<double>> field_03(1, tria, "Scalar Field 03");
  field_03.setup_dofs();
  field_03.setup_vectors();
  field_03.setup_boundary_conditions();
  field_03.close_boundary_conditions(false);
  field_03.apply_boundary_conditions(false);

  const Functions::SquareFunction<dim> function;
  Point<dim> point;
  for (unsigned d=0; d<dim; ++d)
    point[d] = 0.5;
  const double function_value{function.value(point)};

  RMHD::VectorTools::interpolate(field_03,
                                 function,
                                 field_03.solution);
  std::cout << "Difference < 2e-15 ? "
            << std::boolalpha
            << bool(std::fabs(field_03.point_value(point) - function_value) < 2e-15)
            << std::endl;
}



template<int dim>
void test_scalar_field_ptr()
{
  Triangulation<dim>  tria;

  std::shared_ptr<Entities::FE_ScalarField<dim, Vector<double>>> field_01 =
      std::make_shared<Entities::FE_ScalarField<dim, Vector<double>>>(1, tria, "Scalar Field 01");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  // Test setup
  field_01->setup_dofs();
  field_01->setup_vectors();

  std::shared_ptr<Function<dim>>  ptr01 =
      std::make_shared<Functions::ExpFunction<dim>>();
  std::shared_ptr<Function<dim>>  ptr02 =
      std::make_shared<Functions::CutOffFunctionC1<dim>>();

  // Test assignment of boundary conditions
  field_01->setup_boundary_conditions();
  field_01->set_periodic_boundary_condition(0, 1, 0);
  field_01->set_dirichlet_boundary_condition(2, ptr01, true);
  field_01->set_neumann_boundary_condition(3, ptr02);
  field_01->close_boundary_conditions(true);
  field_01->apply_boundary_conditions(true);

  // Test copying and modification of boundary conditions
  std::shared_ptr<Entities::FE_ScalarField<dim, Vector<double>>> field_02 =
      std::make_shared<Entities::FE_ScalarField<dim, Vector<double>>>(*field_01, "Scalar Field 02");
  field_02->setup_dofs();
  field_02->setup_vectors();
  field_02->setup_boundary_conditions();
  Entities::BoundaryConditionsBase<dim> &bc_02 = field_02->get_boundary_conditions();
  const Entities::BoundaryConditionsBase<dim> &bc_01 = field_01->get_boundary_conditions();
  bc_02.copy(bc_01);
  for (auto &dirichlet_bc: bc_02.dirichlet_bcs)
    dirichlet_bc.second = std::make_shared<Functions::ZeroFunction<dim>>();
  const std::vector<types::boundary_id> unconstrained_boundary_ids =
      bc_02.get_unconstrained_boundary_ids();
  for (const auto id: unconstrained_boundary_ids)
    field_02->set_neumann_boundary_condition(id, ptr01, true);
  field_02->close_boundary_conditions(true);
  field_02->apply_boundary_conditions(true);

  // Test point_value
  tria.refine_global(3);
  std::shared_ptr<Entities::FE_ScalarField<dim, Vector<double>>> field_03 =
       std::make_shared<Entities::FE_ScalarField<dim, Vector<double>>>(1, tria, "Scalar Field 03");
  field_03->setup_dofs();
  field_03->setup_vectors();
  field_03->setup_boundary_conditions();
  field_03->close_boundary_conditions(false);
  field_03->apply_boundary_conditions(false);

  const Functions::SquareFunction<dim> function;

  Point<dim> point;
  for (unsigned d=0; d<dim; ++d)
    point[d] = 0.5;
  const double function_value{function.value(point)};

  RMHD::VectorTools::interpolate(*field_03,
                                 function,
                                 field_03->solution);
  std::cout << "Difference < 2e-15 ? "
            << std::boolalpha
            << bool(std::fabs(field_03->point_value(point) - function_value) < 2e-15)
            << std::endl;
}



template<int dim>
void test_vector_field()
{
  Triangulation<dim>  tria;
  Entities::FE_VectorField<dim, Vector<double>> field_01(1, tria, "Vector Field 01");
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  // Test setup
  field_01.setup_dofs();
  field_01.setup_vectors();

  std::shared_ptr<Function<dim>>  ptr01 =
      std::make_shared<Functions::CutOffFunctionC1<dim>>(1.0, Point<dim>(), dim);
  std::shared_ptr<TensorFunction<1, dim>>  ptr02 =
      std::make_shared<ZeroTensorFunction<1, dim>>();
  std::shared_ptr<Function<dim>>  ptr03 =
      std::make_shared<Functions::CosineFunction<dim>>(dim);
  std::shared_ptr<Function<dim>>  ptr04 =
      std::make_shared<Functions::ConstantFunction<dim>>(1.0, dim);

  // Test assignment of boundary conditions
  field_01.setup_boundary_conditions();
  field_01.set_periodic_boundary_condition(0, 1, 0);
  field_01.set_dirichlet_boundary_condition(2, ptr01, true);
  field_01.set_neumann_boundary_condition(3, ptr02);
  if (dim==3)
  {
    field_01.set_tangential_component_boundary_condition(4, ptr03, true);
    field_01.set_normal_component_boundary_condition(5, ptr04, true);
  }
  field_01.close_boundary_conditions(true);
  field_01.apply_boundary_conditions(true);

  // Test copying and modification of boundary conditions
  Entities::FE_VectorField<dim, Vector<double>> field_02(field_01, "Vector Field 02");
  field_02.setup_dofs();
  field_02.setup_vectors();
  field_02.setup_boundary_conditions();
  Entities::BoundaryConditionsBase<dim> &bc_02 = field_02.get_boundary_conditions();
  const Entities::BoundaryConditionsBase<dim> &bc_01 = field_01.get_boundary_conditions();
  bc_02.copy(bc_01);
  for (auto &dirichlet_bc: bc_02.dirichlet_bcs)
    dirichlet_bc.second = std::make_shared<Functions::ZeroFunction<dim>>(dim);
  const std::vector<types::boundary_id> unconstrained_boundary_ids =
      bc_02.get_unconstrained_boundary_ids();
  for (const auto id: unconstrained_boundary_ids)
    field_02.set_neumann_boundary_condition(id, ptr02, true);
  field_02.close_boundary_conditions(true);
  field_02.apply_boundary_conditions(true);

  // Test point_value
  tria.refine_global(3);
  Entities::FE_VectorField<dim, Vector<double>> field_03(2, tria, "Vector Field 03");
  field_03.setup_dofs();
  field_03.setup_vectors();
  field_03.setup_boundary_conditions();
  field_03.close_boundary_conditions(false);
  field_03.apply_boundary_conditions(false);

  const Functions::CosineFunction<dim> function(dim);
  Point<dim> point;
  for (unsigned d=0; d<dim; ++d)
    point[d] = 0.5;
  Vector<double> function_value(function.n_components);
  function.vector_value(point,
                        function_value);

  RMHD::VectorTools::interpolate(field_03,
                                 function,
                                 field_03.solution);
  Tensor<1, dim>  difference;
  for (unsigned int d=0; d<dim; ++d)
    difference[d] = function_value[d];

  difference -= field_03.point_value(point);
  std::cout << "Difference < 2e-15 ? "
            << std::boolalpha
            << bool(difference.norm() < 2e-15)
            << std::endl;
}


template<int dim>
void test_vector_field_ptr()
{
  Triangulation<dim>  tria;

  std::shared_ptr<Entities::FE_VectorField<dim, Vector<double>>> field_01 =
      std::make_shared<Entities::FE_VectorField<dim, Vector<double>>>(1, tria, "Vector Field 01");

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  // Test setup
  field_01->setup_dofs();
  field_01->setup_vectors();

  std::shared_ptr<Function<dim>>  ptr01 =
      std::make_shared<Functions::CutOffFunctionC1<dim>>(1.0, Point<dim>(), dim);
  std::shared_ptr<TensorFunction<1, dim>>  ptr02 =
      std::make_shared<ZeroTensorFunction<1, dim>>();
  std::shared_ptr<Function<dim>>  ptr03 =
      std::make_shared<Functions::CosineFunction<dim>>(dim);
  std::shared_ptr<Function<dim>>  ptr04 =
      std::make_shared<Functions::ConstantFunction<dim>>(1.0, dim);

  // Test assignment of boundary conditions
  field_01->setup_boundary_conditions();
  field_01->set_periodic_boundary_condition(0, 1, 0);
  field_01->set_dirichlet_boundary_condition(2, ptr01, true);
  field_01->set_neumann_boundary_condition(3, ptr02);
  if (dim==3)
  {
    field_01->set_tangential_component_boundary_condition(4, ptr03, true);
    field_01->set_normal_component_boundary_condition(5, ptr04, true);
  }
  field_01->close_boundary_conditions(true);
  field_01->apply_boundary_conditions(true);

  // Test copying and modification of boundary conditions
  std::shared_ptr<Entities::FE_VectorField<dim, Vector<double>>> field_02 =
      std::make_shared<Entities::FE_VectorField<dim, Vector<double>>>(*field_01, "Vector Field 02");
  field_02->setup_dofs();
  field_02->setup_vectors();
  field_02->setup_boundary_conditions();
  Entities::BoundaryConditionsBase<dim> &bc_02 = field_02->get_boundary_conditions();
  const Entities::BoundaryConditionsBase<dim> &bc_01 = field_01->get_boundary_conditions();
  bc_02.copy(bc_01);
  for (auto &dirichlet_bc: bc_02.dirichlet_bcs)
    dirichlet_bc.second = std::make_shared<Functions::ZeroFunction<dim>>(dim);
  const std::vector<types::boundary_id> unconstrained_boundary_ids =
      bc_02.get_unconstrained_boundary_ids();
  for (const auto id: unconstrained_boundary_ids)
    field_02->set_neumann_boundary_condition(id, ptr02, true);
  field_02->close_boundary_conditions(true);
  field_02->apply_boundary_conditions(true);

  // Test point_value
  tria.refine_global(3);
  std::shared_ptr<Entities::FE_VectorField<dim, Vector<double>>> field_03 =
      std::make_shared<Entities::FE_VectorField<dim, Vector<double>>>(2, tria, "Vector Field 03");
  field_03->setup_dofs();
  field_03->setup_vectors();
  field_03->setup_boundary_conditions();
  field_03->close_boundary_conditions(false);
  field_03->apply_boundary_conditions(false);

  const Functions::CosineFunction<dim> function(dim);
  Point<dim> point;
  for (unsigned d=0; d<dim; ++d)
    point[d] = 0.5;
  Vector<double> function_value(function.n_components);
  function.vector_value(point,
                        function_value);

  RMHD::VectorTools::interpolate(*field_03,
                                 function,
                                 field_03->solution);
  Tensor<1, dim>  difference;
  for (unsigned int d=0; d<dim; ++d)
    difference[d] = function_value[d];
  difference -= field_03->point_value(point);
  std::cout << "Difference < 2e-15 ? "
            << std::boolalpha
            << bool(difference.norm() < 2e-15)
            << std::endl;
}



int main(void)
{
  try
  {
    dealii::deallog.depth_console(0);

    test_scalar_field<2>();
    test_scalar_field<3>();

    test_scalar_field_ptr<2>();
    test_scalar_field_ptr<3>();

    test_vector_field<2>();
    test_vector_field<3>();

    test_vector_field_ptr<2>();
    test_vector_field_ptr<3>();
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
