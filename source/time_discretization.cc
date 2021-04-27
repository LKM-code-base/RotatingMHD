#include <rotatingMHD/global.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>

#include <iomanip>
#include <functional>
#include <fstream>
#include <ostream>

namespace RMHD
{

using namespace dealii;

namespace TimeDiscretization
{

TimeDiscretizationParameters::TimeDiscretizationParameters()
:
vsimex_scheme(VSIMEXScheme::CNAB),
n_maximum_steps(100),
adaptive_time_stepping(true),
adaptive_time_step_barrier(2),
initial_time_step(1e-3),
minimum_time_step(1e-9),
maximum_time_step(1e-3),
start_time(0.0),
final_time(1.0),
verbose(false)
{}

TimeDiscretizationParameters::TimeDiscretizationParameters(const std::string &parameter_filename)
:
TimeDiscretizationParameters()
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

void TimeDiscretizationParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Time stepping parameters");
  {

    prm.declare_entry("Time stepping scheme",
                      "CNAB",
                      Patterns::Selection("Euler|CNAB|mCNAB|CNLF|BDF2"));

    prm.declare_entry("Maximum number of time steps",
                      "10",
                      Patterns::Integer());

    prm.declare_entry("Adaptive time stepping",
                      "true",
                      Patterns::Bool());

    prm.declare_entry("Adaptive timestepping barrier",
                      "2",
                      Patterns::Integer());

    prm.declare_entry("Initial time step",
                      "1e-6",
                      Patterns::Double());

    prm.declare_entry("Minimum time step",
                      "1e-6",
                      Patterns::Double());

    prm.declare_entry("Maximum time step",
                      "1e-3",
                      Patterns::Double());

    prm.declare_entry("Start time",
                      "0.0",
                      Patterns::Double(0.));

    prm.declare_entry("Final time",
                      "1.0",
                      Patterns::Double(0.));

    prm.declare_entry("Verbose",
                      "false",
                      Patterns::Bool());
  }
  prm.leave_subsection();
}

void TimeDiscretizationParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Time stepping parameters");
  {
    std::string vsimex_type_str;
    vsimex_type_str = prm.get("Time stepping scheme");

    if (vsimex_type_str == "CNAB")
      vsimex_scheme = VSIMEXScheme::CNAB;
    else if (vsimex_type_str == "mCNAB")
      vsimex_scheme = VSIMEXScheme::mCNAB;
    else if (vsimex_type_str == "CNLF")
      vsimex_scheme = VSIMEXScheme::CNLF;
    else if (vsimex_type_str == "BDF2")
      vsimex_scheme = VSIMEXScheme::BDF2;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected string for variable step size "
                             "IMEX scheme."));

    adaptive_time_stepping = prm.get_bool("Adaptive time stepping");
    if (adaptive_time_stepping)
      adaptive_time_step_barrier = prm.get_integer("Adaptive timestepping barrier");

    Assert(adaptive_time_step_barrier > 0,
           ExcLowerRange(adaptive_time_step_barrier, 0));

    n_maximum_steps = prm.get_integer("Maximum number of time steps");
    Assert(n_maximum_steps > 0, ExcLowerRange(n_maximum_steps, 0));

    initial_time_step = prm.get_double("Initial time step");
    Assert(initial_time_step > 0,
           ExcLowerRangeType<double>(initial_time_step, 0));

    if (adaptive_time_stepping)
    {
        minimum_time_step = prm.get_double("Minimum time step");
        Assert(minimum_time_step > 0,
               ExcLowerRangeType<double>(minimum_time_step, 0));

        maximum_time_step = prm.get_double("Maximum time step");
        Assert(maximum_time_step > 0,
               ExcLowerRangeType<double>(maximum_time_step, 0));

        Assert(minimum_time_step <= maximum_time_step,
               ExcLowerRangeType<double>(minimum_time_step, maximum_time_step));
        Assert(minimum_time_step <= initial_time_step,
               ExcLowerRangeType<double>(minimum_time_step, initial_time_step));
        Assert(initial_time_step <= maximum_time_step,
               ExcLowerRangeType<double>(initial_time_step, maximum_time_step));
    }
    else
    {
        minimum_time_step = 1e-15;

        maximum_time_step = 1e+15;
    }

    start_time = prm.get_double("Start time");
    Assert(start_time >= 0.0, ExcLowerRangeType<double>(start_time, 0.0));

    final_time = prm.get_double("Final time");
    Assert(final_time > 0.0, ExcLowerRangeType<double>(final_time, 0.0));
    Assert(final_time > start_time, ExcLowerRangeType<double>(final_time, start_time));
    Assert(initial_time_step <= final_time,
           ExcLowerRangeType<double>(initial_time_step, final_time));

    verbose = prm.get_bool("Verbose");
  }
  prm.leave_subsection();
}

template<typename Stream>
Stream& operator<<(Stream &stream, const TimeDiscretizationParameters &prm)
{
  const size_t column_width[2] ={ 40, 20 };

  constexpr size_t line_width = 63;

  const char header[] = "+------------------------------------------+"
                        "----------------------+";

  auto add_line = [&]
                  (const char first_column[],
                   const auto second_column)->void
    {
      stream << "| "
             << std::setw(column_width[0]) << first_column
             << " | "
             << std::setw(column_width[1]) << second_column
             << " |"
             << std::endl;
    };

  stream << std::left << header << std::endl;

  stream << "| "
         << std::setw(line_width)
         << "Timestepping parameters"
         << " |"
         << std::endl;

  stream << header << std::endl;

  std::string vsimex_scheme;
  switch (prm.vsimex_scheme)
  {
    case VSIMEXScheme::CNAB:
      vsimex_scheme = "CNAB";
      break;
    case VSIMEXScheme::mCNAB:
      vsimex_scheme = "mCNAB";
      break;
    case VSIMEXScheme::CNLF:
      vsimex_scheme = "CNLF";
      break;
    case VSIMEXScheme::BDF2:
      vsimex_scheme = "BDF2";
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Given VSIMEXScheme is not known or cannot be "
                             "interpreted."));
    break;
  }
  add_line("IMEX scheme", vsimex_scheme);

  add_line("Maximum number of time steps", prm.n_maximum_steps);
  add_line("Adaptive timestepping", (prm.adaptive_time_stepping? "true": "false"));
  if (prm.adaptive_time_stepping)
    add_line("Adaptive timestepping barrier", prm.adaptive_time_step_barrier);
  add_line("Initial time step", prm.initial_time_step);
  if (prm.adaptive_time_stepping)
  {
    add_line("Minimum time step", prm.minimum_time_step);
    add_line("Maximum time step", prm.maximum_time_step);
  }
    add_line("Start time", prm.start_time);
  add_line("Final time", prm.final_time);
  add_line("Verbose", (prm.verbose? "true": "false"));

  stream << header;

  return (stream);
}

template<typename Stream>
Stream& operator<<(Stream &stream, const VSIMEXMethod &vsimex)
{
  stream << "Step = "
         << std::right
         << std::setw(6)
         << vsimex.get_step_number()
         << ","
         << std::left
         << " Current time = "
         << std::scientific
         << vsimex.get_current_time()
         << ","
         << " Next time step = "
         << std::scientific
         << vsimex.get_next_step_size();

  return (stream);
}

template<typename Stream>
void VSIMEXMethod::print_coefficients(Stream &stream, const std::string prefix) const
{
  switch (beta.size())
  {
    case 1:
      stream << prefix.c_str() << "+-------------+------------+------------+\n"
             << prefix.c_str() << "|    Index    |     n      |    n-1     |\n"
             << prefix.c_str() << "+-------------+------------+------------+\n"
             << prefix.c_str() << "|    alpha    | ";
      break;
    case 2:
      stream << prefix.c_str() << "+-------------+------------+------------+------------+\n"
             << prefix.c_str() << "|    Index    |     n      |    n-1     |     n-2    |\n"
             << prefix.c_str() << "+-------------+------------+------------+------------+\n"
             << prefix.c_str() << "|    alpha    | ";
      break;
    default:
      Assert(false,
             ExcMessage("Size of the vector beta does not match the expected "
                        "values."));
      break;
  }

  for (const auto it: alpha)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl
         << prefix.c_str() << "|    beta     |     -      | ";
  for (const auto it: beta)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl
         << prefix.c_str() << "|    gamma    | ";
  for (const auto it: gamma)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl
         << prefix.c_str() << "|  extra_pol  |     -      | ";
  for (const auto it: eta)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl
         << prefix.c_str() << "|  alpha_zero |     -      | ";
  for (const auto it: old_alpha_zero)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl
         << prefix.c_str() << "|old_step_size|     -      | ";
  for (const auto it: old_step_size_values)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }
  stream << std::endl;

  switch (beta.size())
  {
    case 1:
      stream << prefix.c_str() << "+-------------+------------+------------+\n";
      break;
    case 2:
      stream << prefix.c_str() << "+-------------+------------+------------+------------+\n";
      break;

    default:
      Assert(false,
             ExcMessage("Size of the vector beta does not match the expected "
                        "values."));
      break;
  }

  stream << std::fixed << std::setprecision(0);
}

std::string VSIMEXMethod::get_name() const
{
  std::string name;

  switch (parameters.vsimex_scheme)
  {
    case VSIMEXScheme::CNAB:
      name.assign("Crank-Nicolson-Adams-Bashforth");
      break;
    case VSIMEXScheme::mCNAB:
      name.assign("Modified Crank-Nicolson-Adams-Bashforth");
      break;
    case VSIMEXScheme::CNLF:
      name.assign("Crank-Nicolson-Leap-Frog");
      break;
  case VSIMEXScheme::BDF2:
      name.assign("Second order backward differentiation");
      break;
  default:
    AssertThrow(false,
                ExcMessage("Given VSIMEXScheme is not known or cannot be "
                           "interpreted."));
    break;
  }
  return (name);
}

VSIMEXMethod::VSIMEXMethod(const TimeDiscretizationParameters &params)
:
DiscreteTime(params.start_time,
             params.final_time,
             params.initial_time_step),
parameters(params),
omega(1.0),
flag_coefficients_changed(true)
{

  if ((parameters.initial_time_step > parameters.maximum_time_step) ||
      (parameters.initial_time_step < parameters.minimum_time_step))
  {
      std::ostringstream message;
      message << "The desired start step of " << parameters.initial_time_step
              << " is not inside the given bonded range (" << parameters.minimum_time_step
              << ", " << parameters.maximum_time_step << ").";

      AssertThrow(false, ExcMessage(message.str().c_str()));
  }

  switch (parameters.vsimex_scheme)
  {
    case VSIMEXScheme::BDF2 :
      order = 2;
      vsimex_parameters.resize(order);
      vsimex_parameters[0] = 1.0;
      vsimex_parameters[1] = 0.0;
      break;
    case VSIMEXScheme::CNAB :
      order = 2;
      vsimex_parameters.resize(order);
      vsimex_parameters[0] = 0.5;
      vsimex_parameters[1] = 0.0;
      break;
    case VSIMEXScheme::mCNAB :
      order = 2;
      vsimex_parameters.resize(order);
      vsimex_parameters[0] = 0.5;
      vsimex_parameters[1] = 1.0/8.0;
      break;
    case VSIMEXScheme::CNLF :
      order = 2;
      vsimex_parameters.resize(order);
      vsimex_parameters[0] = 0.0;
      vsimex_parameters[1] = 1.0;
      break;
    default:
     Assert(false,
            ExcMessage("Specified scheme is not implemented. See documentation"));
     break;
  };

  reinit();
}

VSIMEXMethod::VSIMEXMethod(const VSIMEXMethod &vsimex)
:
DiscreteTime(vsimex.get_current_time(),
             vsimex.parameters.final_time,
             vsimex.get_next_step_size()),
parameters(vsimex.parameters),
order(vsimex.order),
vsimex_parameters(vsimex.vsimex_parameters),
alpha(vsimex.alpha),
beta(vsimex.beta),
gamma(vsimex.gamma),
eta(vsimex.eta),
omega(vsimex.omega),
old_alpha_zero(vsimex.old_step_size_values),
old_step_size_values(vsimex.old_step_size_values),
flag_coefficients_changed(vsimex.flag_coefficients_changed)
{}

void VSIMEXMethod::clear()
{
  for (unsigned int i=0; i<order+1; ++i)
  {
    alpha[i] = 0.0;
    beta[i] = 0.0;
    gamma[i] = 0.0;
  }

  for (unsigned int i=0; i<order; ++i)
  {
    eta[i] = 0.0;
    old_alpha_zero[i] = 0.0;
    old_step_size_values[i] = 0.0;
  }

  omega = 1.0;

  this->restart();
}

void VSIMEXMethod::reinit()
{
  // Resize all coefficient vectors according to the scheme's order
  // and initializing their values to zero except of those acting as
  // divisor.
  alpha.resize(order+1, 0.0);
  beta.resize(order, 0.0);
  gamma.resize(order+1, 0.0);
  eta.resize(order, 0.0);
  old_step_size_values.resize(order, 0.0);
  old_alpha_zero.resize(order, 1.0);
}

void VSIMEXMethod::set_desired_next_step_size(const double time_step_size)
{
  if (time_step_size < parameters.minimum_time_step)
    DiscreteTime::set_desired_next_step_size(parameters.minimum_time_step);
  else if (time_step_size > parameters.maximum_time_step)
    DiscreteTime::set_desired_next_step_size(parameters.maximum_time_step);
  else
    DiscreteTime::set_desired_next_step_size(time_step_size);
}

void VSIMEXMethod::update_coefficients()
{
  // Computes the ration of the next and previous time step sizes.
  // It is nested in an if as get_previous_step_size() returns zero
  // at the start time.
  if (get_step_number() > 0)
  {
    omega = get_next_step_size() / get_previous_step_size();
    AssertIsFinite(omega);
  }

  // The elimination of the solenoidal velocity in the pressure
  // correction scheme requires to store older values of
  // \f$ \alpha_0\f$ and time steps. The following updates
  // the std::vectors storing those values
  for (unsigned int i = order-1; i > 0; --i)
  {
    old_alpha_zero[i]       = old_alpha_zero[i-1];
    old_step_size_values[i] = old_step_size_values[i-1];
  }

  // and stores their previous values. The inline if considers the
  // change from a first to second order scheme between the first and
  // second time steps.
  old_alpha_zero[0]       = (get_step_number() > 1)
                            ? alpha[0]
                            : (1.0 + 2.0 * vsimex_parameters[0])/2.0;
  old_step_size_values[0] = get_previous_step_size();

  // Checks if the time step size changes. If not, exit the method.
  // The second boolean, i.e. get_step_number() <= (get_order() - 1),
  // takes the first step into account.
  if ((float)omega != 1. || get_step_number() <= (get_order() - 1))
    flag_coefficients_changed = true;
  else
  {
    flag_coefficients_changed = false;
    return;
  }

  // Updates the VSIMEX coefficients. For the first time step, the
  // method returns the coefficients of the Backward Euler scheme instead.
  if (get_step_number() < (get_order() - 1))
  {
    // Hardcoded coefficients for a Backward Euler first order scheme
    const double a = 1.0;

    alpha[0] = 1.0;
    alpha[1] = - 1.0;
    alpha[2] = 0.0;

    beta[0]  = 1.0;
    beta[1]  = 0.0;

    gamma[0] = a;
    gamma[1] = 1.0 - a;
    gamma[2] = 0.0;

    // First order Taylor extrapolation coefficients
    eta[0]   = 1.0;
    eta[1]   = 0.0;
  }
  else
  {
    // VSIMEX coefficient's formulas
    const double a = vsimex_parameters[0];
    const double b = vsimex_parameters[1];

    alpha[0] = (1.0 + 2.0 * a * omega) / (1.0 + omega);
    alpha[1] = ((1.0 - 2.0 * a) * omega - 1.0);
    alpha[2] = (2.0 * a - 1.0) * omega * omega / (1.0 + omega);

    beta[0]  = 1.0 + a * omega;
    beta[1]  = - a * omega;

    gamma[0] = a + b / (2.0 * omega);
    gamma[1] = 1.0 - a - (1.0 + 1.0 / omega) * b / 2.0;
    gamma[2] = b / 2.0;

    // Second order Tayor extrapolation coefficients
    eta[0]   = 1.0 + omega;
    eta[1]   = - omega;
  }

  AssertIsFinite(alpha[0]);
  AssertIsFinite(alpha[1]);
  AssertIsFinite(alpha[2]);
  AssertIsFinite(beta[0]);
  AssertIsFinite(beta[1]);
  AssertIsFinite(gamma[0]);
  AssertIsFinite(gamma[1]);
  AssertIsFinite(gamma[2]);
  AssertIsFinite(eta[0]);
  AssertIsFinite(eta[1]);
}

template<>
void VSIMEXMethod::extrapolate<Vector<double>>
(const Vector<double> &old_values,
 const Vector<double> &old_old_values,
 Vector<double>       &extrapolated_values) const
{
  Assert(old_values.size() == old_old_values.size(),
         ExcDimensionMismatch(old_values.size(), old_old_values.size()));
  Assert(extrapolated_values.size() == old_values.size(),
         ExcDimensionMismatch(extrapolated_values.size(), old_values.size()));

  extrapolated_values = old_values;
  extrapolated_values.sadd(eta[0], eta[1], old_old_values);
}

template<>
void VSIMEXMethod::extrapolate<LinearAlgebra::MPI::Vector>
(const LinearAlgebra::MPI::Vector &old_values,
 const LinearAlgebra::MPI::Vector &old_old_values,
 LinearAlgebra::MPI::Vector       &extrapolated_values) const
{
  Assert(old_values.size() == old_old_values.size(),
         ExcDimensionMismatch(old_values.size(), old_old_values.size()));
  Assert(extrapolated_values.size() == old_values.size(),
         ExcDimensionMismatch(extrapolated_values.size(), old_values.size()));

  extrapolated_values = old_values;
  extrapolated_values.sadd(eta[0], eta[1], old_old_values);
}

template<typename DataType>
void VSIMEXMethod::extrapolate
(const DataType &old_values,
 const DataType &old_old_values,
 DataType       &extrapolated_values) const
{
  extrapolated_values = eta[0] * old_values + eta[1] * old_old_values;
}

template<typename DataType>
void VSIMEXMethod::extrapolate_list
(const std::vector<DataType>  &old_values,
 const std::vector<DataType>  &old_old_values,
 std::vector<DataType>        &extrapolated_values) const
{
  Assert(old_values.size() == old_old_values.size(),
         ExcDimensionMismatch(old_values.size(), old_old_values.size()));
  Assert(extrapolated_values.size() == old_values.size(),
         ExcDimensionMismatch(extrapolated_values.size(), old_values.size()));

  const std::size_t n = extrapolated_values.size();

  for (std::size_t i=0; i<n; ++i)
    extrapolated_values[i] = extrapolate(old_values[i],
                                         old_old_values[i]);
}

} // namespace TimeDiscretiation

} // namespace RMHD

// explicit instantiations
template std::ostream & RMHD::TimeDiscretization::operator<<
(std::ostream &, const RMHD::TimeDiscretization::TimeDiscretizationParameters &);
template dealii::ConditionalOStream & RMHD::TimeDiscretization::operator<<
(dealii::ConditionalOStream &, const RMHD::TimeDiscretization::TimeDiscretizationParameters  &);

template std::ostream & RMHD::TimeDiscretization::operator<<
(std::ostream &, const RMHD::TimeDiscretization::VSIMEXMethod &);
template dealii::ConditionalOStream & RMHD::TimeDiscretization::operator<<
(dealii::ConditionalOStream &, const RMHD::TimeDiscretization::VSIMEXMethod &);

template void RMHD::TimeDiscretization::VSIMEXMethod::print_coefficients
(std::ostream &, const std::string prefix) const;
template void RMHD::TimeDiscretization::VSIMEXMethod::print_coefficients
(dealii::ConditionalOStream &, const std::string prefix) const;

template void RMHD::TimeDiscretization::VSIMEXMethod::extrapolate
(const double &,
 const double &,
 double &) const;

template void RMHD::TimeDiscretization::VSIMEXMethod::extrapolate
(const Tensor<1,2> &,
 const Tensor<1,2> &,
 Tensor<1,2> &) const;

template void RMHD::TimeDiscretization::VSIMEXMethod::extrapolate
(const Tensor<1,3> &,
 const Tensor<1,3> &,
 Tensor<1,3> &) const;

template void RMHD::TimeDiscretization::VSIMEXMethod::extrapolate
(const FEValuesViews::Vector<2>::curl_type &,
 const FEValuesViews::Vector<2>::curl_type &,
 FEValuesViews::Vector<2>::curl_type &) const;

template void RMHD::TimeDiscretization::VSIMEXMethod::extrapolate
(const Tensor<2,2> &,
 const Tensor<2,2> &,
 Tensor<2,2> &) const;

template void RMHD::TimeDiscretization::VSIMEXMethod::extrapolate
(const Tensor<2,3> &,
 const Tensor<2,3> &,
 Tensor<2,3> &) const;
