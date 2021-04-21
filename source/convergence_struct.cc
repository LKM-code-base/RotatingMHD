
#include <rotatingMHD/convergence_struct.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/grid/grid_tools.h>

#include <string>

namespace RMHD
{

namespace ConvergenceTest
{

namespace internal
{
  constexpr char header[] = "+------------------------------------------+"
                        "----------------------+";

  constexpr size_t column_width[2] ={ 40, 20 };

  constexpr size_t line_width = 63;

  template<typename Stream, typename A>
  void add_line(Stream  &stream,
                const A &line)
  {
    stream << "| "
           << std::setw(line_width)
           << line
           << " |"
           << std::endl;
  }

  template<typename Stream, typename A, typename B>
  void add_line(Stream  &stream,
                const A &first_column,
                const B &second_column)
  {
    stream << "| "
           << std::setw(column_width[0]) << first_column
           << " | "
           << std::setw(column_width[1]) << second_column
           << " |"
           << std::endl;
  }

  template<typename Stream>
  void add_header(Stream  &stream)
  {
    stream << std::left << header << std::endl;
  }

} // internal


ConvergenceTestParameters::ConvergenceTestParameters()
:
test_type(ConvergenceTestType::temporal),
n_spatial_cycles(2),
step_size_reduction_factor(0.5),
n_temporal_cycles(2)
{}




ConvergenceTestParameters::ConvergenceTestParameters
(const std::string &parameter_filename)
:
ConvergenceTestParameters()
{
  ParameterHandler prm;

  declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename.c_str());

  if (!parameter_file)
  {
    parameter_file.close();

    std::ostringstream message;
    message << "Input parameter file <"
            << parameter_filename << "> not found. Creating a"
            << std::endl
            << "template file of the same name."
            << std::endl;

    std::ofstream parameter_out(parameter_filename.c_str());
    prm.print_parameters(parameter_out,
                         ParameterHandler::OutputStyle::Text);

    AssertThrow(false, ExcMessage(message.str().c_str()));
  }

  prm.parse_input(parameter_file);

  parse_parameters(prm);
}




void ConvergenceTestParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Convergence test parameters");
  {
    prm.declare_entry("Convergence test type",
                      "temporal",
                      Patterns::Selection("spatial|temporal|both"));

    prm.declare_entry("Number of spatial convergence cycles",
                      "2",
                      Patterns::Integer(1));

    prm.declare_entry("Time-step reduction factor",
                      "0.5",
                      Patterns::Double());

    prm.declare_entry("Number of temporal convergence cycles",
                      "2",
                      Patterns::Integer(1));
  }
  prm.leave_subsection();
}



void ConvergenceTestParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Convergence test parameters");
  {
    if (prm.get("Convergence test type") == std::string("spatial"))
    {
      test_type = ConvergenceTestType::spatial;

      n_spatial_cycles = prm.get_integer("Number of spatial convergence cycles");
      AssertThrow(n_spatial_cycles > 0,
                  ExcLowerRange(n_spatial_cycles, 0));
    }
    else if (prm.get("Convergence test type") == std::string("temporal"))
    {
      test_type = ConvergenceTestType::temporal;

      step_size_reduction_factor = prm.get_double("Time-step reduction factor");
      AssertThrow(step_size_reduction_factor > 0.0,
                  ExcLowerRangeType<double>(step_size_reduction_factor, 0.0));
      AssertThrow(step_size_reduction_factor < 1.0,
                  ExcLowerRangeType<double>(1.0, step_size_reduction_factor));

      n_temporal_cycles = prm.get_integer("Number of temporal convergence cycles");
      AssertThrow(n_temporal_cycles > 0,
                  ExcLowerRange(n_temporal_cycles, 0));
    }
    else if (prm.get("Convergence test type") == std::string("both"))
    {
        test_type = ConvergenceTestType::spatio_temporal;

        step_size_reduction_factor = prm.get_double("Time-step reduction factor");
        AssertThrow(step_size_reduction_factor > 0.0,
                    ExcLowerRangeType<double>(step_size_reduction_factor, 0.0));
        AssertThrow(step_size_reduction_factor < 1.0,
                    ExcLowerRangeType<double>(1.0, step_size_reduction_factor));

        n_temporal_cycles = prm.get_integer("Number of temporal convergence cycles");
        AssertThrow(n_temporal_cycles > 0,
                    ExcLowerRange(n_temporal_cycles, 0));

        n_spatial_cycles = prm.get_integer("Number of spatial convergence cycles");
        AssertThrow(n_spatial_cycles > 0,
                    ExcLowerRange(n_spatial_cycles, 0));
    }
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the type of"
                             " of convergence test."));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const ConvergenceTestParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Convergence test parameters");
  internal::add_header(stream);

  switch (prm.test_type)
  {
    case ConvergenceTestType::spatial:
      internal::add_line(stream, "Convergence test type", "spatial");
      break;
    case ConvergenceTestType::temporal:
      internal::add_line(stream, "Convergence test type", "temporal");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected identifier for the type of"
                               " of convergence test."));
      break;
  }

  internal::add_line(stream,
                     "Number of spatial convergence cycles",
                     prm.n_spatial_cycles);

  internal::add_line(stream,
                     "Number of temporal convergence cycles",
                     prm.n_temporal_cycles);

  internal::add_line(stream,
                     "Time-step reduction factor",
                     prm.step_size_reduction_factor);

  internal::add_header(stream);

  return (stream);
}

ConvergenceTestData::ConvergenceTestData(const ConvergenceTestType &type)
:
type(type),
level(0)
{}

template<int dim, int spacedim>
void ConvergenceTestData::update_table
(const DoFHandler<dim, spacedim> &dof_handler,
 const double step_size,
 const std::map<typename VectorTools::NormType, double> &error_map)
{
  table.add_value("level", level);
  level += 1;

  table.add_value("step size", step_size);

  update_table(dof_handler, error_map);
}


template<int dim, int spacedim>
void ConvergenceTestData::update_table
(const DoFHandler<dim, spacedim> &dof_handler,
 const std::map<typename VectorTools::NormType, double> &error_map)
{
  const types::global_dof_index n_dofs{dof_handler.n_dofs()};
  table.add_value("n_dofs", n_dofs);

  const Triangulation<dim, spacedim> &tria{dof_handler.get_triangulation()};
  const types::global_cell_index n_cells{tria.n_global_active_cells()};
  table.add_value("n_cells", n_cells);

  const double h_max{GridTools::maximal_cell_diameter(tria)};
  table.add_value("h_max", h_max);

  std::string column;
  unsigned int dimension{spacedim};
  bool evaluate_convergence{false};

  switch (type)
  {
    case ConvergenceTestType::spatial:
      column.assign("h_max");
      evaluate_convergence = true;
      break;
    case ConvergenceTestType::temporal:
      column.assign("step size");
      dimension = 1;
      evaluate_convergence = true;
      break;
    case ConvergenceTestType::spatio_temporal:
      break;

    default:
      Assert(false,
             ExcMessage("Unexpected value for the ConvergenceTestType."));
      break;
  }

  std::map<typename VectorTools::NormType, double>::const_iterator it;

  it = error_map.find(VectorTools::NormType::L2_norm);
  if (it != error_map.end())
  {
    table.add_value("L2", it->second);

    if (evaluate_convergence)
      table.evaluate_convergence_rates("L2",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       dimension);
  }

  it = error_map.find(VectorTools::NormType::Linfty_norm);
  if (it != error_map.end())
  {
    table.add_value("Linfty", it->second);

    if (evaluate_convergence)
      table.evaluate_convergence_rates("Linfty",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       dimension);
  }

  it = error_map.find(VectorTools::NormType::H1_norm);
  if (it != error_map.end())
  {
    table.add_value("H1", it->second);

    if (evaluate_convergence)
      table.evaluate_convergence_rates("H1",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       dimension);
  }

}

void ConvergenceTestData::update_table
(const double step_size,
 const std::map<typename VectorTools::NormType, double> &error_map)
{
  table.add_value("level", level);
  level += 1;

  table.add_value("step size", step_size);

  std::string column;
  switch (type)
  {
    case ConvergenceTestType::temporal:
      column.assign("step size");
      break;
    default:
      Assert(false,
             ExcMessage("Wrong case to update_table for the current type of the "
                        "convergence test."));
      break;
  }

  std::map<typename VectorTools::NormType, double>::const_iterator it;

  it = error_map.find(VectorTools::NormType::L2_norm);
  if (it != error_map.end())
  {
    table.add_value("L2", it->second);

    table.evaluate_convergence_rates("L2",
                                     column,
                                     ConvergenceTable::reduction_rate_log2);
  }

  it = error_map.find(VectorTools::NormType::Linfty_norm);
  if (it != error_map.end())
  {
    table.add_value("Linfty", it->second);

    table.evaluate_convergence_rates("Linfty",
                                     column,
                                     ConvergenceTable::reduction_rate_log2);
  }

  it = error_map.find(VectorTools::NormType::H1_norm);
  if (it != error_map.end())
  {
    table.add_value("H1", it->second);

    table.evaluate_convergence_rates("H1",
                                     column,
                                     ConvergenceTable::reduction_rate_log2);
  }
}

void ConvergenceTestData::format_columns()
{
  const unsigned precision{6};

  try
  {
    table.set_scientific("step size", true);
    table.set_precision("step size", 2);
  }
  catch (ConvergenceTable::ExcColumnNotExistent &)
  {
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
      std::abort();
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
      std::abort();
  }

  try
  {
    table.set_scientific("h_max", true);
    table.set_precision("h_max", 2);
  }
  catch (ConvergenceTable::ExcColumnNotExistent &)
  {
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
      std::abort();
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
      std::abort();
  }

  try
  {
    table.set_scientific("L2", true);
    table.set_precision("L2", precision);
    table.add_column_to_supercolumn("L2", "error norms");
  }
  catch (ConvergenceTable::ExcColumnNotExistent &)
  {
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
      std::abort();
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
      std::abort();
  }

  try
  {
    table.set_scientific("H1", true);
    table.set_precision("H1", precision);
    table.add_column_to_supercolumn("L2", "error norms");
  }
  catch (ConvergenceTable::ExcColumnNotExistent &)
  {
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
      std::abort();
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
      std::abort();
  }

  try
  {
      table.set_scientific("Linfty", true);
      table.set_precision("Linfty", precision);
      table.add_column_to_supercolumn("Linfty", "error norms");
  }
  catch (ConvergenceTable::ExcColumnNotExistent &)
  {
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
      std::abort();
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
      std::abort();
  }
}

/*!
 * @brief Output of the convergence table to a stream object,
 */
template<typename Stream>
void ConvergenceTestData::print_data(Stream &stream)
{
  if (level == 0)
    return;

  format_columns();

  table.write_text(stream, TableHandler::org_mode_table);
}

template <>
void ConvergenceTestData::print_data(ConditionalOStream &stream)
{
  if (level == 0)
    return;

  format_columns();

  if (stream.is_active())
    table.write_text(stream.get_stream(), TableHandler::org_mode_table);
}


bool ConvergenceTestData::save(const std::string &file_name)
{
  if (level == 0)
    return (false);

  format_columns();

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::ofstream file(file_name.c_str());
    Assert(file, ExcFileNotOpen(file_name));
    table.write_text(file,
                     TableHandler::TextOutputFormat::org_mode_table);
    file.close();
  }
  return (true);
}

} // namespace ConvergenceTest

} // namespace RMHD


// explicit instantiations
template std::ostream & RMHD::ConvergenceTest::operator<<
(std::ostream &, const RMHD::ConvergenceTest::ConvergenceTestParameters &);
template dealii::ConditionalOStream & RMHD::ConvergenceTest::operator<<
(dealii::ConditionalOStream &, const RMHD::ConvergenceTest::ConvergenceTestParameters &);

template void RMHD::ConvergenceTest::
ConvergenceTestData::update_table<2,2>
(const dealii::DoFHandler<2,2> &,
 const std::map<typename dealii::VectorTools::NormType, double> &);

template void RMHD::ConvergenceTest::
ConvergenceTestData::update_table<3,3>
(const dealii::DoFHandler<3,3> &,
 const std::map<typename dealii::VectorTools::NormType, double> &);

template void RMHD::ConvergenceTest::
ConvergenceTestData::update_table<2,2>
(const dealii::DoFHandler<2,2> &,
 const double,
 const std::map<typename dealii::VectorTools::NormType, double> &);

template void RMHD::ConvergenceTest::
ConvergenceTestData::update_table<3,3>
(const dealii::DoFHandler<3,3> &,
 const double,
 const std::map<typename dealii::VectorTools::NormType, double> &);

template void RMHD::ConvergenceTest::
ConvergenceTestData::print_data
(std::ostream &);
