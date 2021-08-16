#include <rotatingMHD/data_postprocessors.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_spherical.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/math/special_functions/spherical_harmonic.hpp>

#include <fstream>
#include <iostream>

using namespace dealii;


class ScalarFunction01 : public Functions::Spherical<3>
{
public:
  ScalarFunction01(const unsigned int degree, const int order)
  :
  Functions::Spherical<3>(),
  degree(degree),
  order(order)
  {
    Assert(degree >= static_cast<unsigned int>(order), ExcLowerRange(degree, order));
  };

private:

  const unsigned int degree;

  const int order;

  virtual double svalue(const std::array<double, 3> &scoord,
                        const unsigned int /* component = 0 */) const
  {
    if (order < 0)
      return scoord[0] * boost::math::spherical_harmonic_i(degree, -order, scoord[1], scoord[2]);
    else
      return scoord[0] * boost::math::spherical_harmonic_r(degree, order, scoord[1], scoord[2]);
  }
};


template <int dim>
class ScalarFunction02 : public Function<dim>
{
public:
  ScalarFunction02(const unsigned int degree)
  :
  Function<dim>(),
  degree(degree)
  {};

  // access to one component at one point
  double
  value(const Point<dim> &point,
        const unsigned int /* component = 0 */) const
  {
    std::array<double, dim> scoord =
        GeometricUtilities::Coordinates::to_spherical(point);

    return svalue(scoord);
  }

private:

  const unsigned int degree;

  double svalue(const std::array<double, dim> &scoord,
                const unsigned int component = 0) const
  {
    (void) component;
    return std::cos(degree * scoord[dim-1]);
  }
};



template <int dim>
class VectorFunction01 : public Function<dim>
{
public:
  VectorFunction01()
  :
  Function<dim>(dim)
  {};

  void
  vector_value(const Point<dim> &point,
               Vector<double>   &value) const
  {
    std::array<double, dim> scoord =
        GeometricUtilities::Coordinates::to_spherical(point);

    switch (dim)
    {
      case 2:
        value[0] = scoord[0] * cos(scoord[1]) * cos(scoord[1]);
        value[1] = scoord[0] * sin(scoord[1]) * cos(scoord[1]);
        break;
      case 3:
        value[1] = scoord[0] * sin(scoord[1]) * cos(scoord[2]) * cos(scoord[2]);
        value[1] = scoord[0] * sin(scoord[1]) * sin(scoord[1]) * sin(scoord[2]);
        value[2] = scoord[0] * sin(scoord[1]) * cos(scoord[2]) * sin(scoord[2]);
        break;
    }
  }
};


template <int dim>
void postprocessor_scalar_field()
{

  Triangulation<dim>  tria;

  GridGenerator::hyper_cube(tria);
  tria.refine_global(3);

  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler;
  dof_handler.initialize(tria, fe);

  Vector<double>  solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler,
                           Functions::ExpFunction<dim>(),
                           solution);

  RMHD::PostprocessorScalarField<dim> postprocessor("scalar");

  DataOut<dim>  data_out;
  data_out.add_data_vector(dof_handler, solution, postprocessor);
  data_out.build_patches();


  std::stringstream fname;
  fname << "scalar_cartesian_" << dim << "D.vtu";

  std::ofstream  output_file(fname.str().c_str());
  data_out.write_vtu(output_file);

  std::cout << "Output written to " << fname.str() << std::endl;

  return;
}


template <int dim>
void postprocessor_vector_field()
{

  Triangulation<dim>  tria;

  GridGenerator::hyper_cube(tria, -1.0, 1.0);
  tria.refine_global(3);

  FESystem<dim>       fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dof_handler;
  dof_handler.initialize(tria, fe);

  Vector<double>  solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler,
                           Functions::CosineFunction<dim>(dim),
                           solution);

  RMHD::PostprocessorVectorField<dim> postprocessor("vector");

  DataOut<dim>  data_out;
  data_out.add_data_vector(dof_handler, solution, postprocessor);
  data_out.build_patches();


  std::stringstream fname;
  fname << "vector_cartesian_" << dim << "D.vtu";

  std::ofstream  output_file(fname.str().c_str());
  data_out.write_vtu(output_file);

  std::cout << "Output written to " << fname.str() << std::endl;

  return;
}


void polar_postprocessor_scalar_field()
{
  constexpr int dim{2};

  Triangulation<dim>  tria;

  GridGenerator::hyper_ball(tria);
  tria.refine_global(3);

  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler;
  dof_handler.initialize(tria, fe);

  Vector<double>  solution(dof_handler.n_dofs());
  VectorTools::interpolate(MappingQ<dim>(2),
                           dof_handler,
                           ScalarFunction02<dim>(2),
                           solution);

  RMHD::PostprocessorScalarField<dim>           postprocessor("scalar");
  RMHD::SphericalPostprocessorScalarField<dim>  spherical_postprocessor("scalar");

  DataOut<dim>  data_out;
  data_out.add_data_vector(dof_handler, solution, postprocessor);
  data_out.add_data_vector(dof_handler, solution, spherical_postprocessor);
  data_out.build_patches();


  std::stringstream fname;
  fname << "scalar_" << "spherical_" << dim << "D.vtu";

  std::ofstream  output_file(fname.str().c_str());
  data_out.write_vtu(output_file);

  std::cout << "Output written to " << fname.str() << std::endl;

  return;
}


void spherical_postprocessor_scalar_field()
{
  constexpr int dim{3};

  Triangulation<dim>  tria;

  GridGenerator::hyper_ball(tria);
  tria.refine_global(4);

  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler;
  dof_handler.initialize(tria, fe);

  Vector<double>  solution(dof_handler.n_dofs());
  VectorTools::interpolate(MappingQ<dim>(2),
                           dof_handler,
                           ScalarFunction01(3, 2),
                           solution);

  RMHD::PostprocessorScalarField<dim>           postprocessor("scalar");
  RMHD::SphericalPostprocessorScalarField<dim>  spherical_postprocessor("scalar");

  DataOut<dim>  data_out;
  data_out.add_data_vector(dof_handler, solution, postprocessor);
  data_out.add_data_vector(dof_handler, solution, spherical_postprocessor);
  data_out.build_patches();


  std::stringstream fname;
  fname << "scalar_" << "spherical_" << dim << "D.vtu";

  std::ofstream  output_file(fname.str().c_str());
  data_out.write_vtu(output_file);

  std::cout << "Output written to " << fname.str() << std::endl;

  return;
}

template <int dim>
void spherical_postprocessor_vector_field()
{
  Triangulation<dim>  tria;

  GridGenerator::hyper_ball(tria);
  tria.refine_global(4);

  FESystem<dim>   fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dof_handler;
  dof_handler.initialize(tria, fe);

  Vector<double>  solution(dof_handler.n_dofs());
  VectorTools::interpolate(MappingQ<dim>(2),
                           dof_handler,
                           VectorFunction01<dim>(),
                           solution);

  RMHD::PostprocessorVectorField<dim>           postprocessor("scalar");
  RMHD::SphericalPostprocessorVectorField<dim>  spherical_postprocessor("scalar");

  DataOut<dim>  data_out;
  data_out.add_data_vector(dof_handler, solution, postprocessor);
  data_out.add_data_vector(dof_handler, solution, spherical_postprocessor);
  data_out.build_patches();


  std::stringstream fname;
  fname << "vector_" << "spherical_" << dim << "D.vtu";

  std::ofstream  output_file(fname.str().c_str());
  data_out.write_vtu(output_file);

  std::cout << "Output written to " << fname.str() << std::endl;

  return;
}


int main(int /* argc */, char **/* argv */)
{
  try
  {
    dealii::deallog.depth_console(0);

    postprocessor_scalar_field<2>();
    postprocessor_scalar_field<3>();

    postprocessor_vector_field<2>();
    postprocessor_vector_field<3>();

    polar_postprocessor_scalar_field();
    spherical_postprocessor_scalar_field();

    spherical_postprocessor_vector_field<2>();
    spherical_postprocessor_vector_field<3>();

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

