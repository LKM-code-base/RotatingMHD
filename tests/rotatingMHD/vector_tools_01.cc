#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_generator.h>

#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/vector_tools.h>

using namespace dealii;
using namespace RMHD;

// Testing methods inside namespace VectorTools in serial

template<int dim>
void test_vector_field()
{
  Triangulation<dim>            tria;
  Entities::FE_VectorField<dim, Vector<double>> field_01(1, tria, "Vector Field 01");

  std::shared_ptr<Function<dim>>    function_shared_ptr =
      std::make_shared<Functions::ConstantFunction<dim>>(1.0, dim);

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(3);

  field_01.setup_dofs();
  field_01.setup_vectors();

  field_01.setup_boundary_conditions();
  field_01.set_dirichlet_boundary_condition(0, function_shared_ptr);
  field_01.set_dirichlet_boundary_condition(1, function_shared_ptr);
  field_01.set_dirichlet_boundary_condition(2, function_shared_ptr);
  field_01.set_dirichlet_boundary_condition(3, function_shared_ptr);
  if (dim == 3)
  {
    field_01.set_dirichlet_boundary_condition(4, function_shared_ptr);
    field_01.set_dirichlet_boundary_condition(5, function_shared_ptr);
  }

  field_01.close_boundary_conditions(false);
  field_01.apply_boundary_conditions(false);

  RMHD::VectorTools::project(field_01,
                             *function_shared_ptr,
                             field_01.solution);
  RMHD::VectorTools::interpolate(field_01,
                                 *function_shared_ptr,
                                 field_01.old_solution);

  field_01.distributed_vector = field_01.solution;
  field_01.distributed_vector -= field_01.old_solution;

  std::cout << "Vector difference: " << field_01.distributed_vector.l2_norm() << std::endl;

  const std::map<RMHD::VectorTools::NormType, double> errors =
      RMHD::VectorTools::compute_error(field_01, *function_shared_ptr);

  std::cout << "L2 error: " << errors.at(RMHD::VectorTools::NormType::L2_norm) << std::endl;
  std::cout << "H1 error: " << errors.at(RMHD::VectorTools::NormType::H1_norm) << std::endl;
  std::cout << "Linfty error: " << errors.at(RMHD::VectorTools::NormType::Linfty_norm) << std::endl;
}



template<int dim>
void test_scalar_field()
{
  Triangulation<dim>            tria;
  Entities::FE_ScalarField<dim, Vector<double>> field_01(1, tria, "Scalar Field 01");

  std::shared_ptr<Function<dim>>    function_shared_ptr =
      std::make_shared<Functions::ConstantFunction<dim>>(1.0);

  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(3);

  field_01.setup_dofs();
  field_01.setup_vectors();

  field_01.setup_boundary_conditions();
  field_01.set_dirichlet_boundary_condition(0, function_shared_ptr);
  field_01.set_dirichlet_boundary_condition(1, function_shared_ptr);
  field_01.set_dirichlet_boundary_condition(2, function_shared_ptr);
  field_01.set_dirichlet_boundary_condition(3, function_shared_ptr);
  if (dim == 3)
  {
    field_01.set_dirichlet_boundary_condition(4, function_shared_ptr);
    field_01.set_dirichlet_boundary_condition(5, function_shared_ptr);
  }

  field_01.close_boundary_conditions(false);
  field_01.apply_boundary_conditions(false);

  RMHD::VectorTools::project(field_01,
                             *function_shared_ptr,
                             field_01.solution);
  RMHD::VectorTools::interpolate(field_01,
                                 *function_shared_ptr,
                                 field_01.old_solution);

  field_01.distributed_vector = field_01.solution;
  field_01.distributed_vector -= field_01.old_solution;

  std::cout << "Vector difference: " << field_01.distributed_vector.l2_norm() << std::endl;

  const std::map<RMHD::VectorTools::NormType, double> errors =
      RMHD::VectorTools::compute_error(field_01, *function_shared_ptr);

  std::cout << "L2 error: " << errors.at(RMHD::VectorTools::NormType::L2_norm) << std::endl;
  std::cout << "H1 error: " << errors.at(RMHD::VectorTools::NormType::H1_norm) << std::endl;
  std::cout << "Linfty error: " << errors.at(RMHD::VectorTools::NormType::Linfty_norm) << std::endl;
}



int main(void)
{
  try
  {
    dealii::deallog.depth_console(0);

    test_scalar_field<2>();
    test_scalar_field<3>();

    test_vector_field<2>();
    test_vector_field<3>();

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
