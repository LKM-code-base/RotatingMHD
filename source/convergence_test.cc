#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>
#include <rotatingMHD/convergence_test.h>

#include <string.h>

namespace RMHD
{

using namespace dealii;

template <int dim>
ConvergenceAnalysisData<dim>::ConvergenceAnalysisData
(const std::shared_ptr<Entities::EntityBase<dim>> &entity,
 const Function<dim>             &exact_solution)
:
entity(entity),
exact_solution(exact_solution)
{
  convergence_table.declare_column("level");
  convergence_table.declare_column("dt");
  convergence_table.declare_column("cells");
  convergence_table.declare_column("dofs");
  convergence_table.declare_column("hmax");

  convergence_table.declare_column("L2");
  convergence_table.declare_column("H1");
  convergence_table.declare_column("Linfty");

  convergence_table.set_scientific("dt", true);
  convergence_table.set_scientific("hmax", true);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("Linfty", true);

  convergence_table.set_precision("dt", 2);
  convergence_table.set_precision("hmax", 2);
  convergence_table.set_precision("L2", 6);
  convergence_table.set_precision("H1", 6);
  convergence_table.set_precision("Linfty", 6);
}

template <int dim>
void ConvergenceAnalysisData<dim>::update_table
(const unsigned int level,
 const double       time_step,
 const bool         flag_spatial_convergence)
{
  /*
   * Add new entries to the columns describing the spatio-temporal
   * discretization.
   */
  convergence_table.add_value("level", level);
  convergence_table.add_value("dt", time_step);
  convergence_table.add_value("cells", entity->get_triangulation().n_global_active_cells());
  convergence_table.add_value("dofs", entity->dof_handler->n_dofs());
  convergence_table.add_value("hmax", GridTools::maximal_cell_diameter(entity->get_triangulation()));

  {
    // Initialize vector of cell-wise errors
    Vector<double> cellwise_difference(
      entity->get_triangulation().n_active_cells());

    {
      // Compute the error in the L2-norm.
      const QGauss<dim>    quadrature_formula(entity->fe_degree + 2);

      VectorTools::integrate_difference
      (*(entity->dof_handler),
       entity->solution,
       exact_solution,
       cellwise_difference,
       quadrature_formula,
       VectorTools::L2_norm);

      const double L2_error =
        VectorTools::compute_global_error(entity->get_triangulation(),
                                          cellwise_difference,
                                          VectorTools::L2_norm);
      convergence_table.add_value("L2", L2_error);

      // Compute the error in the H1-norm.
      VectorTools::integrate_difference
      (*(entity->dof_handler),
       entity->solution,
       exact_solution,
       cellwise_difference,
       quadrature_formula,
       VectorTools::H1_norm);

      const double H1_error =
        VectorTools::compute_global_error(entity->get_triangulation(),
                                          cellwise_difference,
                                          VectorTools::H1_norm);
      convergence_table.add_value("H1", H1_error);
    }

    /*
     * For the infinity norm, the quadrature rule is designed such that the
     * quadrature points coincide with the support points. For a polynomial degree
     * less than three, the approach pursued here works, because the support
     * points of the finite element discretization are equidistantly spaced on
     * the reference element. Otherwise, the quadrature might be inappropriate.
     *
     * In other words, we simply sample the solution at the support points, i.e.,
     * the nodes.
     *
     */
    {
      const QTrapez<1>     trapezoidal_rule;
      const QIterated<dim> linfty_quadrature_rule(trapezoidal_rule,
                                                  entity->fe_degree);

      // Compute the error in the Linfty-norm.
      VectorTools::integrate_difference
      (*entity->dof_handler,
       entity->solution,
       exact_solution,
       cellwise_difference,
       linfty_quadrature_rule,
       VectorTools::Linfty_norm);

      const double Linfty_error =
        VectorTools::compute_global_error(entity->get_triangulation(),
                                          cellwise_difference,
                                          VectorTools::Linfty_norm);

      convergence_table.add_value("Linfty", Linfty_error);
    }
  }

  // Compute convergence rates
  const std::string reference_column = (flag_spatial_convergence) ?
                                  "hmax" : "dt";

  convergence_table.evaluate_convergence_rates
  ("L2",
   reference_column,
   ConvergenceTable::reduction_rate_log2,
   1);

  convergence_table.evaluate_convergence_rates
  ("H1",
   reference_column,
   ConvergenceTable::reduction_rate_log2,
   1);
  convergence_table.evaluate_convergence_rates

  ("Linfty",
   reference_column,
   ConvergenceTable::reduction_rate_log2,
   1);
}

template<typename Stream, int dim>
Stream& operator<<(Stream &stream,
                   const ConvergenceAnalysisData<dim> &data)
{
  stream << std::endl;
  stream << "                               " << data.entity->name
         << " convergence table" << std::endl
         << "==============================================="
         << "==============================================="
         << std::endl;

  std::ostringstream aux_stream;

  data.convergence_table.write_text(aux_stream);

  stream << aux_stream.str().c_str();

  return (stream);
}

template <int dim>
void ConvergenceAnalysisData<dim>::write_text(std::string filename) const
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    const std::string suffix(".txt");

    // Check if the suffix of the filename matches the desired suffix
    if (std::equal(suffix.rbegin(), suffix.rend(), filename.rbegin()) == false)
    {
      if (filename.find_last_of(".") == filename.size())
        filename.append("txt");
      else
        filename.append(suffix);
    }

    std::ofstream file(filename);

    convergence_table.write_text(file,
                                 TableHandler::TextOutputFormat::org_mode_table);
  }
}

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

        AssertThrow(false, ExcNotImplemented());
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
{
  table.declare_column("level");
}

template<int dim, int spacedim>
void ConvergenceTestData::update_table
(const DoFHandler<dim, spacedim> &dof_handler,
 const double step_size,
 const std::map<typename VectorTools::NormType, double> &error_map)
{
  table.add_value("step size", step_size);
  step_size_specified = true;

  update_table(dof_handler, error_map);
}


template<int dim, int spacedim>
void ConvergenceTestData::update_table
(const DoFHandler<dim, spacedim> &dof_handler,
 const std::map<typename VectorTools::NormType, double> &error_map)
{
  table.add_value("level", level);
  level += 1;

  const types::global_dof_index n_dofs{dof_handler.n_dofs()};
  table.add_value("n_dofs", n_dofs);

  const Triangulation<dim, spacedim> &tria{dof_handler.get_triangulation()};
  const types::global_cell_index n_cells{tria.n_global_active_cells()};
  table.add_value("n_cells", n_cells);


  const double h_max{GridTools::maximal_cell_diameter(tria)};
  table.add_value("h_max", h_max);
  h_max_specified = true;

  std::map<typename VectorTools::NormType, double>::const_iterator it;

  it = error_map.find(VectorTools::NormType::L2_norm);
  if (it != error_map.end())
  {
    table.add_value("L2", it->second);
    L2_error_specified = true;
  }

  it = error_map.find(VectorTools::NormType::Linfty_norm);
  if (it != error_map.end())
  {
    table.add_value("Linfty", it->second);
    Linfty_error_specified = true;
  }

  it = error_map.find(VectorTools::NormType::H1_norm);
  if (it != error_map.end())
  {
    table.add_value("H1", it->second);
    H1_error_specified = true;
  }

}

void ConvergenceTestData::update_table
(const double step_size,
 const std::map<typename VectorTools::NormType, double> &error_map)
{
  table.add_value("level", level);
  level += 1;

  table.add_value("step size", step_size);
  step_size_specified = true;

  std::map<typename VectorTools::NormType, double>::const_iterator it;

  it = error_map.find(VectorTools::NormType::L2_norm);
  if (it != error_map.end())
  {
    table.add_value("L2", it->second);
    L2_error_specified = true;
  }

  it = error_map.find(VectorTools::NormType::Linfty_norm);
  if (it != error_map.end())
  {
    table.add_value("Linfty", it->second);
    Linfty_error_specified = true;
  }

  it = error_map.find(VectorTools::NormType::H1_norm);
  if (it != error_map.end())
  {
    table.add_value("H1", it->second);
    H1_error_specified = true;
  }
}

void ConvergenceTestData::format_columns()
{
  constexpr unsigned int precision{6  };

  if (step_size_specified)
  {
      table.set_scientific("step size", true);
      table.set_precision("step size", 2);
  }

  if (h_max_specified)
  {
    table.set_scientific("h_max", true);
    table.set_precision("h_max", 2);
  }

  if (L2_error_specified)
  {
    table.set_scientific("L2", true);
    table.set_precision("L2", precision);
  }

  if (H1_error_specified)
  {
    table.set_scientific("H1", true);
    table.set_precision("H1", precision);
  }

  if (Linfty_error_specified)
  {
    table.set_scientific("Linfty", true);
    table.set_precision("Linfty", precision);
  }


  std::string column;
  bool evaluate_convergence{false};

  switch (type)
  {
    case ConvergenceTestType::spatial:
      column.assign("h_max");
      evaluate_convergence = true;
      break;
    case ConvergenceTestType::temporal:
      column.assign("step size");
      evaluate_convergence = true;
      break;
    case ConvergenceTestType::spatio_temporal:
      // Do not evaluate convergence rates for a spatio-temporal test,
      // leaving switch-case to avoid throwing an exception.
      break;
    default:
      Assert(false,
             ExcMessage("Unexpected value for the ConvergenceTestType."));
      break;
  }
  if (evaluate_convergence)
	{
    if (L2_error_specified)
      table.evaluate_convergence_rates("L2",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       1);
    if (Linfty_error_specified)
      table.evaluate_convergence_rates("Linfty",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       1);
    if (H1_error_specified)
      table.evaluate_convergence_rates("H1",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       1);
	}
}

template<typename Stream>
Stream& operator<<(Stream &stream, ConvergenceTestData &data)
{
  if (data.level == 0)
    return (stream);

  data.format_columns();

  data.table.write_text(stream);

  return (stream);
}

template <>
ConditionalOStream& operator<<(ConditionalOStream &stream, ConvergenceTestData &data)
{
  if (data.level == 0)
    return (stream);

  data.format_columns();

  if (stream.is_active())
  	data.table.write_text(stream.get_stream());

  return (stream);
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
template struct RMHD::ConvergenceAnalysisData<2>;
template struct RMHD::ConvergenceAnalysisData<3>;

template std::ostream& RMHD::operator<<
(std::ostream &, const RMHD::ConvergenceAnalysisData<2> &);
template std::ostream& RMHD::operator<<
(std::ostream &, const RMHD::ConvergenceAnalysisData<3> &);

template dealii::ConditionalOStream& RMHD::operator<<
(dealii::ConditionalOStream &, const RMHD::ConvergenceAnalysisData<2> &);
template dealii::ConditionalOStream& RMHD::operator<<
(dealii::ConditionalOStream &, const RMHD::ConvergenceAnalysisData<3> &);

template std::ostream& RMHD::ConvergenceTest::operator<<
(std::ostream &, const RMHD::ConvergenceTest::ConvergenceTestParameters &);
template dealii::ConditionalOStream& RMHD::ConvergenceTest::operator<<
(dealii::ConditionalOStream &, const RMHD::ConvergenceTest::ConvergenceTestParameters &);

template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<1, 1> &,
 const double ,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<1, 2> &,
 const double ,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<2, 2> &,
 const double ,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<1, 3> &,
 const double ,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<2, 3> &,
 const double ,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<3, 3> &,
 const double ,
 const std::map<typename dealii::VectorTools::NormType, double> &);

template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<1, 1> &,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<1, 2> &,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<2, 2> &,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<1, 3> &,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<2, 3> &,
 const std::map<typename dealii::VectorTools::NormType, double> &);
template
void RMHD::ConvergenceTest::ConvergenceTestData::update_table
(const dealii::DoFHandler<3, 3> &,
 const std::map<typename dealii::VectorTools::NormType, double> &);

template std::ostream& RMHD::ConvergenceTest::operator<<
(std::ostream &, RMHD::ConvergenceTest::ConvergenceTestData &);
