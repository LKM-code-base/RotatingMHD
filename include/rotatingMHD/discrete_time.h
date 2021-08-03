#ifndef INCLUDE_ROTATINGMHD_DISCRETE_TIME_H_
#define INCLUDE_ROTATINGMHD_DISCRETE_TIME_H_

namespace RMHD
{

namespace TimeDiscretization
{

class DiscreteTime
{
public:
  DiscreteTime(const double start_time,
               const double end_time,
               const double desired_start_step_size = 0.);

  double
  get_current_time() const;

  double
  get_next_time() const;

  double
  get_previous_time() const;

  double
  get_start_time() const;

  double
  get_end_time() const;

  bool
  is_at_start() const;

  bool
  is_at_end() const;

  double
  get_next_step_size() const;

  double
  get_previous_step_size() const;

  unsigned int
  get_step_number() const;

  void
  set_desired_next_step_size(const double time_step_size);

  void
  advance_time();

  void
  restart();

  void
  set_end_time(const double new_end_time);

  /*!
   * @brief Output of the current step number, the current time and the size of
   * the time step.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const DiscreteTime &time);

  private:
  double start_time;

  double end_time;

  double current_time;

  double next_time;

  double previous_time;

  double start_step_size;

  unsigned int step_number;
};

/*!
 * @brief Output of the current step number, the current time and the size of
 * the time step.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const DiscreteTime &time);


/*---------------------- Inline functions ------------------------------*/


inline double
DiscreteTime::get_start_time() const
{
 return (start_time);
}



inline double
DiscreteTime::get_end_time() const
{
 return (end_time);
}



inline bool
DiscreteTime::is_at_start() const
{
 return (step_number == 0);
}



inline bool
DiscreteTime::is_at_end() const
{
 return (current_time == end_time);
}



inline double
DiscreteTime::get_next_step_size() const
{
 return (next_time - current_time);
}



inline double
DiscreteTime::get_previous_step_size() const
{
 return (current_time - previous_time);
}



inline double
DiscreteTime::get_current_time() const
{
 return (current_time);
}



inline double
DiscreteTime::get_next_time() const
{
 return (next_time);
}



inline double
DiscreteTime::get_previous_time() const
{
 return (previous_time);
}



inline unsigned int
DiscreteTime::get_step_number() const
{
 return (step_number);
}

} // namespace TimeDiscretization

} // namespace RMHD


#endif /* INCLUDE_ROTATINGMHD_DISCRETE_TIME_H_ */
