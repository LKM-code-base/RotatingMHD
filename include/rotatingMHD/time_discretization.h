
#ifndef INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_
#define INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_

#include <deal.II/base/discrete_time.h>

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
  ForwardEuler,     /**< Applies the explicit Euler method. */
  CNFE,   /**< Applies Crank-Nicolson to \f$ g(u) \f$ and forward Euler to \f$ f(u) \f$.*/
  /*
   * The following enum is duplicate!?
   */
  BEFE,   /**< Applies Crank-Nicolson to \f$ g(u) \f$ and forward Euler to \f$ f(u) \f$. */
  BDF2,   /**< Applies the backward differentiation formula of second order. */
  CNAB,   /**< Applies Crank-Nicolson to \f$ g(u) \f$ and Adams-Bashforth to \f$ f(u) \f$. */
  mCNAB,  /**< Applies the modified CNAB method. */
  CNLF    /**< Applies Crank-Nicolson to \f$ g(u) \f$ and Leap-Frog to \f$ f(u) \f$. */
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
  *  @brief A method to output the coefficients to the terminal.
  */
  template<typename Stream>
  void output(Stream &stream) const;
};

/*!
* @class VSIMEXMethod
* @brief A time stepping class implementing the VSIMEX coefficients.
* @details Here goes a longer explination with formulas of the VSIMEX
* general scheme.
*/
class VSIMEXMethod : public DiscreteTime
{

public:
  /*!
  * @brief The constructor of the class.
  */
  VSIMEXMethod(const VSIMEXScheme         scheme,
               const double               start_time,
               const double               end_time,
               const double               desired_start_step_size,
               const double               timestep_lower_bound,
               const double               timestep_upper_bound);
// Should I update this constructor for a bounded time step or delete it?
  VSIMEXMethod(const unsigned int         order,
               const std::vector<double>  parameters,
               const double               start_time,
               const double               end_time,
               const double               desired_start_step_size = 0);

  /*!
  * @brief A method returning the order of the VSIMEX scheme.
  */
  unsigned int        get_order();

  /*!
  * @brief A method returning the parameters of the VSIMEX scheme.
  */
  std::vector<double> get_parameters();

  /*!
  * @brief A method to get the updated coefficients.
  * @details This method calls a private method which computes the
  * updated coefficients and then passes the values to the output.
  * @attention The method has to be called between 
  * the set_new_time_step() and the advance_time() methods in order for
  * it to calculate the correct parameters.
  */ 
  void                get_coefficients(VSIMEXCoefficients &output);

  /*!
  * @brief A method passing the proposed new timestep to the class.
  * @details The method checks if the the timestep is inside the bounds
  * set in the constructor. If not, it adjust the timestep accordingly
  * and passes it to the set_desired_time_step() method from the
  * DiscreteTime class which does further modifications if needed.
  */
  void                set_proposed_step_size(const double &timestep);

private:

  VSIMEXScheme              scheme;               /**< VSIMEX scheme being used. */

  unsigned int              order;                /**< Order of the VSIMEX scheme. */

  std::vector<double>       parameters;           /**< Parameters of the VSIMEX scheme. */

  VSIMEXCoefficients        coefficients;         /**< Coefficients of the VSIMEX scheme. */

  double                    omega;                /**< Step-size ratio. */
  double                    timestep;             /**< Current timestep. */
  double                    old_timestep;         /**< Previous timestep. */
  double                    timestep_lower_bound; /**< Lower bound of the timestep. */
  double                    timestep_upper_bound; /**< Upper bound of the timestep. */

  /*!
  * @brief A method that updates the coefficients.
  * @details Here goes a longer explination with the formulas
  */
  void update_coefficients();
};

} // namespace TimeDiscretization
} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_ */
