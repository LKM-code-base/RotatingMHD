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
adaptive_time_stepping(true),
adaptive_time_step_barrier(2),
initial_time_step(1e-3),
minimum_time_step(1e-9),
maximum_time_step(1e-3),
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

    prm.declare_entry("adaptive_time_stepping",
                      "true",
                      Patterns::Bool(),
                      "Turn adaptive time stepping on or off");

    prm.declare_entry("adaptive_timestep_barrier",
                      "2",
                      Patterns::Integer(),
                      "Time step after which adaptive time stepping is applied.");

    prm.declare_entry("initial_time_step",
                      "1e-6",
                      Patterns::Double(),
                      "Size of the initial time step.");

    prm.declare_entry("minimum_time_step",
                      "1e-6",
                      Patterns::Double(),
                      "Size of the minimum time step.");

    prm.declare_entry("maximum_time_step",
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
    else if (vsimex_type_str == "BDF2")
      vsimex_scheme = VSIMEXScheme::BDF2;
    else if (vsimex_type_str == "Euler")
      vsimex_scheme = VSIMEXScheme::ForwardEuler;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected string for variable step size "
                             "IMEX scheme."));

    adaptive_time_stepping = prm.get_bool("adaptive_time_stepping");
    if (adaptive_time_stepping)
      adaptive_time_step_barrier = prm.get_integer("adaptive_timestep_barrier");
    Assert(adaptive_time_step_barrier > 0,
           ExcLowerRange(adaptive_time_step_barrier, 0));

    n_maximum_steps = prm.get_integer("n_maximum_steps");
    Assert(n_maximum_steps > 0, ExcLowerRange(n_maximum_steps, 0));

    initial_time_step = prm.get_double("initial_time_step");
    Assert(initial_time_step > 0,
           ExcLowerRangeType<double>(initial_time_step, 0));

    if (adaptive_time_stepping)
    {
        minimum_time_step = prm.get_double("minimum_time_step");
        Assert(minimum_time_step > 0,
               ExcLowerRangeType<double>(minimum_time_step, 0));

        maximum_time_step = prm.get_double("maximum_time_step");
        Assert(maximum_time_step > 0,
               ExcLowerRangeType<double>(maximum_time_step, 0));

        Assert(minimum_time_step < maximum_time_step,
               ExcLowerRangeType<double>(minimum_time_step, maximum_time_step));
        Assert(minimum_time_step <= initial_time_step,
               ExcLowerRangeType<double>(minimum_time_step, initial_time_step));
        Assert(initial_time_step <= maximum_time_step,
               ExcLowerRangeType<double>(initial_time_step, maximum_time_step));
    }

    start_time = prm.get_double("start_time");
    Assert(start_time >= 0.0, ExcLowerRangeType<double>(start_time, 0.0));

    final_time = prm.get_double("final_time");
    Assert(final_time > 0.0, ExcLowerRangeType<double>(final_time, 0.0));
    Assert(final_time > start_time, ExcLowerRangeType<double>(final_time, start_time));
    Assert(initial_time_step <= final_time,
           ExcLowerRangeType<double>(initial_time_step, final_time));

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
         << "   adaptive_timestep: " << (adaptive_time_stepping? "true": "false") << std::endl
         << "   adaptive_timestep_barrier: " << adaptive_time_step_barrier << std::endl
         << "   initial_timestep: " << initial_time_step << std::endl
         << "   minimum_timestep: " << minimum_time_step << std::endl
         << "   maximum_timestep: " << maximum_time_step << std::endl
         << "   start_time: " << start_time << std::endl
         << "   final_time: " << final_time << std::endl
         << "   verbose: " << (verbose? "true": "false") << std::endl;
}

void VSIMEXMethod::reinit()
{
  alpha.resize(order+1, 0.0);
  beta.resize(order, 0.0);
  gamma.resize(order+1, 0.0);
}

template<typename Stream>
Stream& operator<<(Stream &stream, const VSIMEXMethod &vsimex)
{
  switch (vsimex.beta.size())
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

  for (const auto it: vsimex.alpha)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl << "|    beta     |     -      | ";
  for (const auto it: vsimex.beta)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl << "|    gamma    | ";
  for (const auto it: vsimex.gamma)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }

  stream << std::endl << "|  extra_pol  |     -      | ";
  for (const auto it: vsimex.phi)
  {
    stream << std::setw(10)
           << std::setprecision(2)
           << std::scientific
           << std::right
           << it;
    stream << " | ";
  }
  stream << std::endl;

  switch (vsimex.beta.size())
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

  return stream;
}

VSIMEXMethod::VSIMEXMethod(const TimeSteppingParameters &params)
:
DiscreteTime(params.start_time,
             params.final_time,
             params.initial_time_step),
parameters(params),
omega(1.0),
time_step(parameters.initial_time_step),
old_time_step(parameters.initial_time_step)
{
  Assert(((time_step > parameters.maximum_time_step) &&
          (time_step < parameters.minimum_time_step)),
         ExcMessage("The desired start step is not inside the given bonded range."));

  switch (parameters.vsimex_scheme)
  {
    case VSIMEXScheme::ForwardEuler :
      order = 1;
      imex_constants.resize(order);
      imex_constants[0] = 0.;
      break;
    case VSIMEXScheme::CNFE :
      order = 1;
      imex_constants.resize(order);
      imex_constants[0] = 0.5;
      break;
    case VSIMEXScheme::BEFE :
      order = 1;
      imex_constants.resize(order);
      imex_constants[0] = 1.0;
      break;
    case VSIMEXScheme::BDF2 :
      order = 2;
      imex_constants.resize(order);
      imex_constants[0] = 1.0;
      imex_constants[1] = 0.0;
      break;
    case VSIMEXScheme::CNAB :
      order = 2;
      imex_constants.resize(order);
      imex_constants[0] = 0.5;
      imex_constants[1] = 0.0;
      break;
    case VSIMEXScheme::mCNAB :
      order = 2;
      imex_constants.resize(order);
      imex_constants[0] = 0.5;
      imex_constants[1] = 1.0/8.0;
      break;
    case VSIMEXScheme::CNLF :
      order = 2;
      imex_constants.resize(order);
      imex_constants[0] = 0.0;
      imex_constants[1] = 1.0;
      break;
    default:
     Assert(false,
            ExcMessage("Specified scheme is not implemented. See documentation"));
     break;
  };

  reinit();
}

void VSIMEXMethod::set_desired_next_step_size(const double time_step_size)
{
  if (time_step_size < parameters.minimum_time_step)
    DiscreteTime::set_desired_next_step_size(parameters.minimum_time_step);
  else if (time_step_size > parameters.maximum_time_step)
    DiscreteTime::set_desired_next_step_size(parameters.maximum_time_step);
  else
    DiscreteTime::set_desired_next_step_size(time_step);
}

void VSIMEXMethod::update_coefficients()
{
  time_step      = get_next_step_size();

  old_time_step  = get_previous_step_size();

  omega         = time_step / old_time_step;

  switch (order)
  {
    case 1 :
    {
      const double a   = imex_constants[0];

      alpha[0] = - 1.0 / time_step;
      alpha[1] = 1.0 / time_step;

      beta[0]  = 1.0;

      gamma[0] = ( 1.0 - a);
      gamma[1] = a;

      /*
       * These values are definitely wrong in case of variable size
       * of the time step!
       */
      phi[0]   = -1.0;
      phi[1]   = 2.0;

      break;
    }
    case 2 :
    {
      const double a = imex_constants[0];
      const double b = imex_constants[1];

      alpha[0] = (2.0 * a - 1.0) * omega * omega /
                 (1.0 + omega) / time_step;
      alpha[1] = ((1.0 - 2.0 * a) * omega - 1.0) / time_step;
      alpha[2] = (1.0 + 2.0 * a * omega) /
                 (1.0 + omega) / time_step;

      beta[0]  = - a * omega;
      beta[1]  = 1.0 + a * omega;

      gamma[0] = b / 2.0;
      gamma[1] = 1.0 - a - (1.0 + 1.0 / omega) * b / 2.0;
      gamma[2] = a + b / (2.0 * omega);

      /*
       * These values seem to be correct in case of variable size
       * of the time step!
       */
      phi[0]   = - omega;
      phi[1]   = 1.0 + omega;

      break;
    }
    default :
     Assert(false,
            ExcMessage("Only VSIMEX of first and second order are currently implemented"));
     break;
  }
}

} // namespace TimeDiscretiation

} // namespace RMHD

// explicit instantiations
template void RMHD::TimeDiscretization::TimeSteppingParameters::write(std::ostream &) const;
template void RMHD::TimeDiscretization::TimeSteppingParameters::write(dealii::ConditionalOStream &) const;

template std::ostream & RMHD::TimeDiscretization::operator<<
(std::ostream &, const RMHD::TimeDiscretization::VSIMEXMethod &);
template dealii::ConditionalOStream & RMHD::TimeDiscretization::operator<<
(dealii::ConditionalOStream &, const RMHD::TimeDiscretization::VSIMEXMethod &);
