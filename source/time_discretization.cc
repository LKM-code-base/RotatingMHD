#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/conditional_ostream.h>

#include <fstream>
#include <ostream>

namespace RMHD
{

using namespace dealii;

namespace TimeDiscretization
{

TimeSteppingParameters::TimeSteppingParameters()
:
vsimex_scheme(VSIMEXScheme::CNAB),
n_maximum_steps(100),
adaptive_timestep(true),
adaptive_timestep_barrier(2),
initial_timestep(1e-3),
minimum_timestep(1e-9),
maximum_timestep(1e-3),
start_time(0.0),
final_time(1.0),
verbose(false)
{}

TimeSteppingParameters::TimeSteppingParameters(const std::string &parameter_filename)
:
TimeSteppingParameters()
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

void TimeSteppingParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Time stepping settings");
  {
    prm.declare_entry("time_stepping_scheme",
                      "CNAB",
                      Patterns::Selection("Euler|CNAB|mCNAB|CNLF|BDF2"),
                      "Time stepping scheme applied.");

    prm.declare_entry("n_maximum_steps",
                      "10",
                      Patterns::Integer(),
                      "Maximum number of time steps to be computed.");

    prm.declare_entry("adaptive_timestep",
                      "true",
                      Patterns::Bool(),
                      "Turn adaptive time stepping on or off");

    prm.declare_entry("adaptive_timestep_barrier",
                      "2",
                      Patterns::Integer(),
                      "Time step after which adaptive time stepping is applied.");

    prm.declare_entry("initial_timestep",
                      "1e-6",
                      Patterns::Double(),
                      "Size of the initial time step.");

    prm.declare_entry("minimum_timestep",
                      "1e-6",
                      Patterns::Double(),
                      "Size of the minimum time step.");

    prm.declare_entry("maximum_timestep",
                      "1e-3",
                      Patterns::Double(),
                      "Size of the maximum time step.");

    prm.declare_entry("start_time",
                      "0.0",
                      Patterns::Double(0.),
                      "Start time of the simulation.");

    prm.declare_entry("final_time",
                      "1.0",
                      Patterns::Double(0.),
                      "Final time of the simulation.");

    prm.declare_entry("verbose",
                      "false",
                      Patterns::Bool(),
                      "Activate verbose output.");
}
prm.leave_subsection();
}

void TimeSteppingParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Time stepping settings");
  {
    std::string vsimex_type_str;
    vsimex_type_str = prm.get("time_stepping_scheme");

    if (vsimex_type_str == "CNAB")
      vsimex_scheme = VSIMEXScheme::CNAB;
    else if (vsimex_type_str == "MCNAB")
      vsimex_scheme = VSIMEXScheme::mCNAB;
    else if (vsimex_type_str == "CNLF")
      vsimex_scheme = VSIMEXScheme::CNLF;
    else if (vsimex_type_str == "SBDF")
      vsimex_scheme = VSIMEXScheme::BDF2;
    else if (vsimex_type_str == "Euler")
      vsimex_scheme = VSIMEXScheme::ForwardEuler;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected string for variable step size "
                             "IMEX scheme."));

    adaptive_timestep = prm.get_bool("adaptive_timestep");
    if (adaptive_timestep)
      adaptive_timestep_barrier = prm.get_integer("adaptive_timestep_barrier");
    Assert(adaptive_timestep_barrier > 0,
           ExcLowerRange(adaptive_timestep_barrier, 0));

    n_maximum_steps = prm.get_integer("n_maximum_steps");
    Assert(n_maximum_steps > 0, ExcLowerRange(n_maximum_steps, 0));

    initial_timestep = prm.get_double("initial_timestep");
    Assert(initial_timestep > 0,
           ExcLowerRangeType<double>(initial_timestep, 0));

    if (adaptive_timestep)
    {
        minimum_timestep = prm.get_double("minimum_timestep");
        Assert(minimum_timestep > 0,
               ExcLowerRangeType<double>(minimum_timestep, 0));

        maximum_timestep = prm.get_double("maximum_timestep");
        Assert(maximum_timestep > 0,
               ExcLowerRangeType<double>(maximum_timestep, 0));

        Assert(minimum_timestep < maximum_timestep,
               ExcLowerRangeType<double>(minimum_timestep, maximum_timestep));
        Assert(minimum_timestep <= initial_timestep,
               ExcLowerRangeType<double>(minimum_timestep, initial_timestep));
        Assert(initial_timestep <= maximum_timestep,
               ExcLowerRangeType<double>(initial_timestep, maximum_timestep));
    }

    start_time = prm.get_double("start_time");
    Assert(start_time >= 0.0, ExcLowerRangeType<double>(start_time, 0.0));

    final_time = prm.get_double("final_time");
    Assert(final_time > 0.0, ExcLowerRangeType<double>(final_time, 0.0));
    Assert(final_time > start_time, ExcLowerRangeType<double>(final_time, start_time));
    Assert(initial_timestep <= final_time,
           ExcLowerRangeType<double>(initial_timestep, final_time));

    verbose = prm.get_bool("verbose");
  }
  prm.leave_subsection();

}

template<typename Stream>
void TimeSteppingParameters::write(Stream &stream) const
{
  stream << "Time stepping parameters" << std::endl
         << "   imex_scheme: ";
  switch (vsimex_scheme)
  {
  case VSIMEXScheme::ForwardEuler:
       stream << "Euler" << std::endl;
      break;
  case VSIMEXScheme::CNAB:
      stream << "CNAB" << std::endl;
      break;
  case VSIMEXScheme::mCNAB:
      stream << "MCNAB" << std::endl;
      break;
  case VSIMEXScheme::CNLF:
      stream << "CNLF" << std::endl;
      break;
  case VSIMEXScheme::BDF2:
      stream << "BDF2" << std::endl;
      break;
  default:
    AssertThrow(false,
                ExcMessage("Given VSIMEXScheme is not known or cannot be "
                           "interpreted."));
    break;
  }
  stream << "   n_maximum_steps: " << n_maximum_steps << std::endl
         << "   adaptive_timestep: " << (adaptive_timestep? "true": "false") << std::endl
         << "   adaptive_timestep_barrier: " << adaptive_timestep_barrier << std::endl
         << "   initial_timestep: " << initial_timestep << std::endl
         << "   minimum_timestep: " << minimum_timestep << std::endl
         << "   maximum_timestep: " << maximum_timestep << std::endl
         << "   start_time: " << start_time << std::endl
         << "   final_time: " << final_time << std::endl
         << "   verbose: " << (verbose? "true": "false") << std::endl;
}

VSIMEXCoefficients::VSIMEXCoefficients()
{}

VSIMEXCoefficients::VSIMEXCoefficients(const unsigned int &order)
:
alpha(order+1),
beta(order),
gamma(order+1),
phi(2)
{
  Assert( ( ( order == 1) || ( order == 2 ) ),
         ExcMessage("Only VSIMEX of first and second order are currently "
                    "implemented"));
}

VSIMEXCoefficients::VSIMEXCoefficients(const VSIMEXCoefficients &data)
:
alpha(data.alpha),
beta(data.beta),
gamma(data.gamma),
phi(data.phi)
{}

VSIMEXCoefficients VSIMEXCoefficients::operator=(const VSIMEXCoefficients &data_to_copy)
{
  this->alpha = data_to_copy.alpha;
  this->beta  = data_to_copy.beta;
  this->gamma = data_to_copy.gamma;
  this->phi = data_to_copy.phi;
  return *this;
}

void VSIMEXCoefficients::reinit(const unsigned int order)
{
  *this = VSIMEXCoefficients(order);
}



template<typename Stream>
void VSIMEXCoefficients::output(Stream &stream) const
{
  switch (beta.size())
  {
    case 1:
      stream << "+-------------+------------+------------+\n"
             << "|    Index    |     n      |    n-1     |\n"
             << "+-------------+------------+------------+\n"
             << "|    alpha    | ";
      break;
    case 2:
      stream << "+-------------+------------+------------+------------+\n"
             << "|    Index    |     n      |    n-1     |     n-2    |\n"
             << "+-------------+------------+------------+------------+\n"
             << "|    alpha    | ";
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

  stream << std::endl << "|    beta     |     -      | ";
  for (const auto it: beta)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl << "|    gamma    | ";
  for (const auto it: gamma)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl << "|  extra_pol  |     -      | ";
  for (const auto it: phi)
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
      stream << "+-------------+------------+------------+\n";
      break;
    case 2:
      stream << "+-------------+------------+------------+------------+\n";
      break;

    default:
      Assert(false,
             ExcMessage("Size of the vector beta does not match the expected "
                        "values."));
      break;
  }

  stream << std::fixed << std::setprecision(0);
}

VSIMEXMethod::VSIMEXMethod
(const VSIMEXScheme         scheme,
 const double               start_time,
 const double               end_time,
 const double               desired_start_step_size,
 const double               timestep_lower_bound,
 const double               timestep_upper_bound)
:
DiscreteTime(start_time, end_time, desired_start_step_size),
scheme(scheme),
omega(1.0),
timestep(desired_start_step_size),
old_timestep(desired_start_step_size),
timestep_lower_bound(timestep_lower_bound),
timestep_upper_bound(timestep_upper_bound)
{
  Assert(((timestep > timestep_lower_bound) && 
          (timestep < timestep_upper_bound)),
      ExcMessage(
        "The desired start step is not inside the give bounded range."));
  switch (scheme)
  {
    case VSIMEXScheme::ForwardEuler :
      order = 1;
      parameters.resize(order);
      parameters[0] = 0.;
      break;
    case VSIMEXScheme::CNFE :
      order = 1;
      parameters.resize(order);
      parameters[0] = 0.5;
      break;
    case VSIMEXScheme::BEFE :
      order = 1;
      parameters.resize(order);
      parameters[0] = 1.0;
      break;
    case VSIMEXScheme::BDF2 :
      order = 2;
      parameters.resize(order);
      parameters[0] = 1.0;
      parameters[1] = 0.0;
      break;
    case VSIMEXScheme::CNAB :
      order = 2;
      parameters.resize(order);
      parameters[0] = 0.5;
      parameters[1] = 0.0;
      break;
    case VSIMEXScheme::mCNAB :
      order = 2;
      parameters.resize(order);
      parameters[0] = 0.5;
      parameters[1] = 1.0/8.0;
      break;
    case VSIMEXScheme::CNLF :
      order = 2;
      parameters.resize(order);
      parameters[0] = 0.0;
      parameters[1] = 1.0;
      break;
    default:
     Assert(false,
      ExcMessage("Specified scheme is not implemented. See documentation"));
  };
  coefficients.reinit(order);
}

VSIMEXMethod::VSIMEXMethod
(const unsigned int         order,
 const std::vector<double>  parameters,
 const double               start_time,
 const double               end_time,
 const double               desired_start_step_size)
:
DiscreteTime(start_time, end_time, desired_start_step_size),
order(order),
parameters(parameters),
coefficients(order),
omega(1.0),
timestep(desired_start_step_size),
old_timestep(desired_start_step_size)
{
  Assert( ( ( order == 1) || ( order == 2 )),
    ExcMessage("Only VSIMEX of first and second order are currently implemented"));
}

unsigned int VSIMEXMethod::get_order()
{
  return order;
}

std::vector<double> VSIMEXMethod::get_parameters()
{
  return parameters;
}

void VSIMEXMethod::get_coefficients(VSIMEXCoefficients &output)
{
  update_coefficients();
  output = coefficients;
}

void VSIMEXMethod::set_proposed_step_size(const double &timestep)
{
  if (timestep < timestep_lower_bound)
    set_desired_next_step_size(timestep_lower_bound);
  else if (timestep > timestep_upper_bound)
    set_desired_next_step_size(timestep_upper_bound);
  else
    set_desired_next_step_size(timestep);
}

void VSIMEXMethod::update_coefficients()
{
  timestep      = get_next_step_size();
  old_timestep  = get_previous_step_size();
  omega         = timestep / old_timestep;

  switch (order)
  {
    case 1 :
    {
      static const double gamma   = parameters[0];
      coefficients.alpha[0] = - 1.0 / timestep;
      coefficients.alpha[1] = 1.0 / timestep;
      coefficients.beta[0]  = 1.0;
      coefficients.gamma[0] = ( 1.0 - gamma);
      coefficients.gamma[1] = gamma;
      coefficients.phi[0]   = -1.0;
      coefficients.phi[1]   = 2.0;
      break;
    }
    case 2 :
    {
      static const double gamma   = parameters[0];
      static const double c       = parameters[1];
      coefficients.alpha[0] = (2.0 * gamma - 1.0) * omega * omega / 
                                (1.0 + omega) / timestep;
      coefficients.alpha[1] = ((1.0 - 2.0 * gamma) * omega - 1.0) / timestep;
      coefficients.alpha[2] = (1.0 + 2.0 * gamma * omega) / 
                                (1.0 + omega) / timestep;
      coefficients.beta[0]  = - gamma * omega;
      coefficients.beta[1]  = 1.0 + gamma * omega;
      coefficients.gamma[0] = c / 2.0;
      coefficients.gamma[1] = 1.0 - gamma - (1.0 + 1.0 / omega) * c / 2.0; 
      coefficients.gamma[2] = gamma + c / (2.0 * omega); 
      coefficients.phi[0]   = - omega;
      coefficients.phi[1]   = 1.0 + omega;
      break;
    }
    default :
     Assert(false,
            ExcMessage("Only VSIMEX of first and second order are currently implemented"));
     break;
  }
  //coefficients.output();
}

} // namespace TimeDiscretiation

} // namespace RMHD

// explicit instantiations
template void RMHD::TimeDiscretization::TimeSteppingParameters::write(std::ostream &) const;
template void RMHD::TimeDiscretization::TimeSteppingParameters::write(dealii::ConditionalOStream &) const;

template void RMHD::TimeDiscretization::VSIMEXCoefficients::output(std::ostream &) const;
template void RMHD::TimeDiscretization::VSIMEXCoefficients::output(dealii::ConditionalOStream &) const;
