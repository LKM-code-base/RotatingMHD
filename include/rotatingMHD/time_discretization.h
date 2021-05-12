
#ifndef INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_
#define INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_

#include <rotatingMHD/discrete_time.h>

#include <deal.II/base/parameter_handler.h>

#include <iostream>
#include <vector>

namespace RMHD
{

namespace TimeDiscretization
{

/*!
 * @enum VSIMEXScheme
 * @brief An enum class for the different VSIMEX schemes.
 */
enum class VSIMEXScheme
{
  /*!
   * @brief The second order backward differentiation formula.
   */
  BDF2,
  /*!
   * @brief The classical Crank-Nicolson-Adams-Bashforth scheme.
   * @details Applies Crank-Nicolson to \f$ g(u) \f$ and Adams-Bashforth extrapolation
   * to \f$ f(u) \f$.
   */
  CNAB,
  /*!
   * @brief The modified Crank-Nicolson-Adams-Bashforth scheme.
   */
  mCNAB,
  /*!
   * @brief The Crank-Nicolson-Leap-Frog scheme.
   * @details Applies Crank-Nicolson to \f$ g(u) \f$ and Leap-Frog to
   * \f$f(u)\f$.
   */
  CNLF
};



/*!
 * @struct TimeDiscretizationParameters
 *
 * @brief This structure manages the parameters of the time stepping scheme and
 * is used to control the behavior of VSIMEXMethod.
 */
struct TimeDiscretizationParameters
{

  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  TimeDiscretizationParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  TimeDiscretizationParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the time stepping scheme from
   * the ParameterHandler object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const TimeDiscretizationParameters &prm);

  /*!
   * @brief Type of variable step-size IMEX scheme which is applied.
   */
  VSIMEXScheme  vsimex_scheme;

  /*!
   * @brief Maximum number of timesteps to be performed.
   */
  unsigned int  n_maximum_steps;

  /*!
   * @brief Boolean flag to enable an adaptive adjustment of the size of the
   * time step.
   */
  bool          adaptive_time_stepping;

  /*!
   * @brief Number of the time step from which the adaptive adjustment of the
   * size of the time step is enabled.
   */
  unsigned int  adaptive_time_step_barrier;

  /*!
   * @brief Size of the initial time step.
   */
  double        initial_time_step;

  /*!
   * @brief Size of the maximum time step.
   */
  double        minimum_time_step;

  /*!
   * @brief Size of the minimum time step.
   */
  double        maximum_time_step;

  /*!
   * @brief Time at which the simulation starts.
   */
  double        start_time;

  /*!
   * @brief Time at which the simulation should terminate.
   */
  double        final_time;

  /*!
   * @brief Boolean flag to enable verbose output of @ref VSIMEXMethod.
   */
  bool          verbose;
};



/*!
 * @brief Method forwarding the parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const TimeDiscretizationParameters &prm);



/*!
* @class VSIMEXMethod
*
* @brief A time stepping class implementing the VSIMEX coefficients.
*
* @details Here goes a longer explanation with formulas of the VSIMEX
* general scheme.
*/
class VSIMEXMethod : public DiscreteTime
{

public:

  /*!
  * @brief The constructor of the class.
  */
  VSIMEXMethod(const TimeDiscretizationParameters &parameters);

  /*!
  * @brief Copy constructor.
  */
  VSIMEXMethod(const VSIMEXMethod &other);

  /*!
   * @brief Release all memory and reset all objects.
   */
  void clear();

  /*!
   * @brief Initialize all coefficients of the schemes based on the value of @p previous_step_size.
   */
  void initialize(const double previous_step_size);

  /*!
  * @brief A method returning the order of the VSIMEX scheme.
  */
  unsigned int get_order() const;

  /*!
   * @brief Returns a string with the name of the variable step size IMEX
   * scheme.
   */
  std::string get_name() const;

  /*!
   * @brief A method returning the coefficients \f$\alpha_i \f$.
   */
  const std::vector<double>& get_alpha() const;

  /*!
   * @brief A method returning the coefficients \f$\beta_i \f$.
   */
  const std::vector<double>& get_beta() const;

  /*!
   * @brief A method returning the coefficients \f$\gamma_i \f$.
   */
  const std::vector<double>& get_gamma() const;

  /*!
   * @brief A method returning the coefficients \f$\phi_i \f$.
   */
  const std::vector<double>& get_eta() const;

  /*!
   * @brief A method returning the coefficients \f$ \alpha_0 \f$ of the previous
   * time steps.
   */
  const std::vector<double>& get_previous_alpha_zeros() const;

  /*!
   * @brief A method returning the sizes \f$ \Delta_t \f$ of the previous time
   * steps.
   */
  const std::vector<double>& get_previous_step_sizes() const;

  /*!
   * @brief A method returning the minimum size of the time step.
   */
  double get_minimum_step_size() const;

  /*!
   * @brief A method returning the maximum size of the time step.
   */
  double get_maximum_step_size() const;

  /*!
  * @brief A method setting the *desired* minimum size of the time step.
  *
  * @details This method assigns @p step_size to @ref mininum_step_size.
  *
  */
  void set_minimum_step_size(const double step_size);

  /*!
  * @brief A method setting the *desired* maximum size of the time step.
  *
  * @details This method assigns @p step_size to @ref maximum_step_size.
  *
  */
  void set_maximum_step_size(const double step_size);

  /*!
  * @brief A method passing the *desired* size of the next time step to the
  * class.
  * @details The method checks if the the time step is inside the bounds
  * set in the constructor. If not, it adjusts the time step accordingly
  * and passes it to the set_desired_time_step() method from the
  * DiscreteTime class which does further modifications if needed.
  */
  void set_desired_next_step_size(const double time_step_size);

  template<typename DataType>
  DataType extrapolate(const DataType &old_values,
                       const DataType &old_old_values) const;

  template<typename DataType>
  void extrapolate(const DataType &old_values,
                   const DataType &old_old_values,
                   DataType       &extrapolated_values) const;

  template<typename DataType>
  void extrapolate_list(const std::vector<DataType> &old_values,
                        const std::vector<DataType> &old_old_values,
                        std::vector<DataType>       &extrapolated_values) const;

  /*!
   * @brief Output of the current step number, the current time and the size of
   * the time step.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const VSIMEXMethod &vsimex);

  /*!
   * @brief Output of the current table of coefficients of the variable step
   * size IMEX scheme to a stream object.
   */
  template<typename Stream>
  void print_coefficients(Stream &stream, const std::string prefix="") const;

  /*!
  *  @brief A method that updates the coefficients.
  *  @details Here goes a longer explanation with the formulas.
  */
  void update_coefficients();

  /*!
   * @brief Returns the flag indicating if the VSIMEX coefficients changed from
   * the last step.
   * @details The flag is set as true if @ref omega is anything other
   * than 1.0 and is set as false if @ref is equal to 1.0.
   */
  bool coefficients_changed() const;

private:
  /*!
   * @brief Order of the VSIMEX scheme.
   */
  constexpr static unsigned int order{2};

  /*!
   * @brief Method which updates the sizes of the coefficient vectors .
   */
  void reinit();

  /*
   * @brief Type of the VSIMEX scheme.
   */
  const VSIMEXScheme  type;

  /*!
   * @brief Parameters of the VSIMEX scheme.
   * @attention This designation is very misleading w.r.t. the
   * TimeDiscretizationParameters!
   */
  std::vector<double> vsimex_parameters;

  /*!
   * @brief A vector containing the \f$ \alpha \f$ coefficients.
   * @details Public access is provided through the method get_alpha().
   */
  std::vector<double> alpha;

  /*!
   * @brief A vector containing the \f$ \beta \f$ coefficients.
   * @details Public access is provided through the method get_beta().
   */
  std::vector<double> beta;

  /*!
   * @brief A vector with the \f$ \gamma \f$ coefficients.
   * @details Public access is provided through the method get_gamma().
   */
  std::vector<double> gamma;

  /*!
   * @brief A vector containing coefficients required for extrapolation.
   */
  std::vector<double> eta;

  /*!
   * @brief Ratio of the sizes of the current and the old time step. Denoted by
   * \f$\omega\f$.
   * @details The ratio is given by \f$\omega=\frac{\Delta t_n}{\Delta t_{n-1}}\f$.
   */
  double              omega;

  /*!
   * @brief A vector containing the \f$ \alpha_0 \f$ of the previous time steps.
   * @attention This member is only useful in the NavierStokesProjection
   * class.
   */
  std::vector<double> previous_alpha_zeros;

  /*!
   * @brief A vector containing the sizes of the previous time steps.
   * @details The DiscreteTime class stores only the previous time step.
   * This member stores \f$ n \f$ time steps prior to it, where \f$ n \f$
   * is the order of the scheme.
   * @attention This member is only useful in the NavierStokesProjection
   * class.
   */
  std::vector<double> previous_step_sizes;

  /*!
   * @brief The minimum size of the time step.
   */
  double  minimum_step_size;

  /*!
   * @brief The maximum size of the time step.
   */
  double  maximum_step_size;

  /*!
   * @brief A flag indicating whether the VSIMEX coefficients changed in
   * the last step.
   *
   * @details The flag is set as true if @ref omega is anything other
   * than unity and is set as false if @ref is equal to unity.
   */
  bool    flag_coefficients_changed;

  /*!
   * @brief A flag indicating whether the VSIMEX scheme is restarted.
   *
   * @details The flag is set in @ref initialize and prevents a modification
   * of the coefficients in the next call to @ref update_coefficients.
   */
  bool    flag_restart;

};



/*!
 * @brief Output of the current step number, the current time and the size of
 * the time step.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const VSIMEXMethod &vsimex);



// inline functions
inline unsigned int VSIMEXMethod::get_order() const
{
  return (order);
}

inline const std::vector<double>& VSIMEXMethod::get_alpha() const
{
  return (alpha);
}

inline const std::vector<double>& VSIMEXMethod::get_beta() const
{
  return (beta);
}

inline const std::vector<double>& VSIMEXMethod::get_gamma() const
{
  return (gamma);
}

inline const std::vector<double>& VSIMEXMethod::get_eta() const
{
  return (eta);
}

inline const std::vector<double>& VSIMEXMethod::get_previous_alpha_zeros() const
{
  return (previous_alpha_zeros);
}

inline const std::vector<double>& VSIMEXMethod::get_previous_step_sizes() const
{
  return (previous_step_sizes);
}

inline bool VSIMEXMethod::coefficients_changed() const
{
  return (flag_coefficients_changed);
}

inline double VSIMEXMethod::get_minimum_step_size() const
{
  return (minimum_step_size);
}

inline double VSIMEXMethod::get_maximum_step_size() const
{
  return (maximum_step_size);
}

inline void VSIMEXMethod::set_minimum_step_size(const double step_size)
{
  if (is_at_end())
    return;

  using namespace dealii;

  AssertThrow(maximum_step_size >= step_size,
              ExcLowerRangeType<double>(maximum_step_size, step_size));
  AssertThrow(get_next_step_size() >= step_size,
              ExcLowerRangeType<double>(get_next_step_size(), step_size));

  minimum_step_size = step_size;
}

inline void VSIMEXMethod::set_maximum_step_size(const double step_size)
{
  if (is_at_end())
    return;

  using namespace dealii;

  AssertThrow(step_size >= get_next_step_size(),
              ExcLowerRangeType<double>(step_size, get_next_step_size()));
  AssertThrow(step_size >= minimum_step_size,
              ExcLowerRangeType<double>(step_size, minimum_step_size));

  maximum_step_size = step_size;
}
} // namespace TimeDiscretization

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_ */
