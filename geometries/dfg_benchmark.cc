#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/parameter_handler.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

class ParameterReader : public Subscriptor
{
public:
  unsigned int  refinements;
  double        shell_region_width;
  unsigned int  n_shells;
  double        skewness;

  ParameterReader()
  {}
  void read_parameters(const std::string &);
private:
  void              declare_parameters();
  ParameterHandler  prm;
};

void ParameterReader::declare_parameters()
{
prm.declare_entry("refinements",
                  "2",
                  Patterns::Integer(0),
                  "refinements");
prm.declare_entry("shell_region_width",
                  "0.03",
                  Patterns::Double(0),
                  "shell_region_width" );
prm.declare_entry("n_shells",
                  "2",
                  Patterns::Integer(0),
                  "n_shells" );
prm.declare_entry("skewness",
                  "2.0",
                  Patterns::Double(0),
                  "skewness" );
}

void ParameterReader::read_parameters(const std::string &parameter_file)
{
  declare_parameters();
  prm.parse_input(parameter_file);

  refinements         = prm.get_integer("refinements");
  shell_region_width  = prm.get_double("shell_region_width");
  n_shells            = prm.get_integer("n_shells");
  skewness            = prm.get_double("skewness");
}



int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, 1);

  ParameterReader     prm;
  prm.read_parameters("prm.prm");

  //Triangulation<2>    triangulation

  parallel::distributed::Triangulation<2>    triangulation(MPI_COMM_WORLD,
                  typename Triangulation<2>::MeshSmoothing(
                  Triangulation<2>::smoothing_on_refinement |
                  Triangulation<2>::smoothing_on_coarsening));

  GridGenerator::channel_with_cylinder(triangulation,
                                       prm.shell_region_width,
                                       prm.n_shells,
                                       prm.skewness,
                                       true);
  triangulation.refine_global(prm.refinements);
  GridTools::scale(10.0, triangulation);
  {
    std::ofstream out("dfg.inp");
    GridOut       grid_out;
    GridOutFlags::Ucd ucd_flags(false, true, true);
    grid_out.set_flags(ucd_flags);
    grid_out.write_ucd(triangulation, out);
    std::cout << "Grid written to dfg.inp" << std::endl;
  }
  {
    std::ofstream out("dfg.svg");
    GridOut       grid_out;
    grid_out.write_svg(triangulation, out);
    std::cout << "Grid written to dfg.svg" << std::endl;
  }
  std::cout << "Number of active cells                = " 
            << triangulation.n_active_cells() << std::endl;

  DoFHandler<2> dof_handler(triangulation);
  FE_Q<2>       fe(1);
  dof_handler.distribute_dofs(fe);

  Vector<double>  fake_data;
  fake_data.reinit(dof_handler.n_dofs());

  fake_data = 0.;
  DataOut<2>        data_out;

  data_out.add_data_vector(dof_handler, 
                           fake_data, 
                           "FakeData");
  data_out.build_patches();
  
  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);
}