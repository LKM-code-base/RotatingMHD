
#ifndef INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_
#define INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/parameter_handler.h>

#include <iostream>
#include <vector>

namespace RMHD
{

using namespace dealii;

namespace TimeDiscretization
{

/*!
 * @enum VSIMEXScheme
 * @brief An enum class for the different VSIMEX schemes.
 */
enum class VSIMEXScheme
{
  /*!
   * Forward Euler method.
   */
  ForwardEuler,
  /*!
   * @brief Combination of the Crank-Nicolson and forward Euler method.
   * @details Applies Crank-Nicolson to \f$ g(u) \f$ and forward Euler to \f$ f(u) \f$.
   * @attention SG: What is meant by \f$ f(u) \f$ and \f$ g(u) \f$? What is the
   * wealth of combining both schemes?
   */
  CNFE,
  /*!
   * @brief Combination of the Crank-Nicolson and forward (backward?) Euler method.
   * @details Applies Crank-Nicolson to \f$ g(u) \f$ and forward Euler to \f$ f(u) \f$.
   * @attention SG: The following enum is duplicate!? What is the difference to
   * the previous one?
   */
  BEFE,
  /*!
   * @brief Applies the backward differentiation formula of second order.
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
 * @struct TimeSteppingParameters
 * @brief This structure manages the parameters of the time stepping scheme and
 * is used to control the behavior of VSIMEXMethod.
 */
struct TimeSteppingParameters
{

  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  TimeSteppingParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  TimeSteppingParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the time stepping scheme from
   * the ParameterHandler object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   */
  template<typename Stream>
  void write(Stream &stream) const;

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
  bool          adaptive_time_step;

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
   * @brief Boolean flag to enable verbose output of VSIMEXMethod.
   */
  bool          verbose;
};

/*!
* @struct VSIMEXCoefficients
* @brief An struct containing all the coefficients of the VSIMEX schemes.
* @details The struct also includes the coefficients of a Taylor
* extrapolation of first order.
*/
struct VSIMEXCoefficients
{
  std::vector<double> alpha;  /**< A vector with the \f$ \alpha \f$ coefficients. */

  std::vector<double> beta;   /**< A vector with the \f$ \beta \f$ coefficients. */

  std::vector<double> gamma;  /**< A vector with the \f$ \gamma \f$ coefficients. */

  std::vector<double> phi;    /**< A vector with the Taylor expansion coefficients. */

  /*!
  *  @brief The default constructor of the struct.
  */
  VSIMEXCoefficients();

  /*!
  *  @brief A constructor taking the order of the scheme as input.
  */
  VSIMEXCoefficients(const unsigned int                 &order);

  /*!
  *  @brief A copy constructor.
  */
  VSIMEXCoefficients(const VSIMEXCoefficients           &data);

  /*!
  *  @brief Overloaded operator used in the copy constructor.
  */
  VSIMEXCoefficients operator=(const VSIMEXCoefficients &data_to_copy);

  /*!
  *  @brief A reinitializing method.
  *  @details Passes the order to the respective constructor.
  */
  void reinit(const unsigned int order);

  /*!
  *  A method to output the coefficients to the terminal.
  */
  template<typename Stream>
  void output(Stream &stream) const;
};

/*!
* @class VSIMEXMethod
* @brief A time stepping class implementing the VSIMEX coefficients.
* @details Here goes a longer explanation with formulas of the VSIMEX
* general scheme.
*/
class VSIMEXMethod : public DiscreteTime
{

public:

  /*!
  * @brief The constructor of the class.
  */
  VSIMEXMethod(const TimeSteppingParameters &parameters);

  /*!
  * @brief A method returning the order of the VSIMEX scheme.
  */
  unsigned int get_order() const;

  /*!
  * @brief A method returning the parameters of the VSIMEX scheme.
  */
  std::vector<double> get_parameters() const;

  /*!
  * @brief A method to get the updated coefficients.
  * @details This method calls a private method which computes the
  * updated coefficients and then passes the values to the output.
  * @attention The method has to be called between 
  * the set_new_time_step() and the advance_time() methods in order for
  * it to calculate the correct parameters.
  * @attention In the final version of VSIMEXMethod, we must make sure that the
  * method is called in the right place and otherwise an error is thrown.The
  * *stupid user* might be aware of this constraint!
  */ 
  void get_coefficients(VSIMEXCoefficients &output);

  /*!
  * @brief A method passing the *desired* size of the next time step to the
  * class.
  * @details The method checks if the the time step is inside the bounds
  * set in the constructor. If not, it adjusts the time step accordingly
  * and passes it to the set_desired_time_step() method from the
  * DiscreteTime class which does further modifications if needed.
  */
  void set_desired_next_step_size(const double time_step_size);

private:

  /*!
   * @brief VSIMEX scheme being used.
   */
  VSIMEXScheme        scheme;

  /*!
   * @brief Order of the VSIMEX scheme.
   */
  unsigned int        order;

  /*!
   * @brief Parameters of the VSIMEX scheme.
   * @attention This designation is very misleading w.r.t. the
   * TimeSteppingParameters!
   */
  std::vector<double> imex_constants;

  /*!
   * @brief Coefficients of the VSIMEX scheme.
   */
  VSIMEXCoefficients  coefficients;

  /*!
   * @brief Ratio of the sizes of the current and the old time step. Denoted by
   * \f$\omega\f$.
   * @details The ratio is given by \f$\omega=\frac{\Delta t_n}{\Delta t_{n-1}}\f$.
   */
  double              omega;

  /*!
   * @brief Size of the current time step \f$\Delta t_n\f$.
   * @attention Is duplicate because this variable exist the parent class DiscreteTime!
   */
  double              time_step;

  /*!
   * @brief Size of the previous time step \f$ \Delta t_{n-1}\f$.
   * @attention Is duplicate because this variable exist the parent class DiscreteTime!
   */
  double              old_time_step;

  double              timestep_lower_bound; /**< Lower bound of the timestep. */
  double              timestep_upper_bound; /**< Upper bound of the timestep. */

  /*!
  *  @brief A method that updates the coefficients.
  *  @details Here goes a longer explanation with the formulas.
  */
  void update_coefficients();
};

// inline functions
inline unsigned int VSIMEXMethod::get_order() const
{
  return order;
}

inline std::vector<double> VSIMEXMethod::get_parameters() const
{
  return imex_constants;
}

} // namespace TimeDiscretization

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_ */
