
#include <rotatingMHD/discrete_time.h>

#include <deal.II/base/exceptions.h>

namespace RMHD
{

namespace TimeDiscretization
{

namespace
{
   // Helper function that computes the next discrete time, adjusting it if:
   //  - The next time exceeds the end time.
   //  - The next time is smaller but very close to the end time.
   double
   calculate_next_time(const double current_time,
                       const double step_size,
                       const double end_time)
   {
     Assert(step_size >= 0., dealii::ExcMessage("Time step size must be non-negative"));
     Assert(end_time >= current_time, dealii::ExcInternalError());
     double           next_time          = current_time + step_size;
     constexpr double relative_tolerance = 0.05;
     const double     time_tolerance     = relative_tolerance * step_size;
     if (next_time > end_time - time_tolerance)
       next_time = end_time;
     return next_time;
   }
} // namespace



 DiscreteTime::DiscreteTime(const double start_time,
                            const double end_time,
                            const double desired_start_step_size)
   : start_time{start_time}
   , end_time{end_time}
   , current_time{start_time}
   , next_time{calculate_next_time(start_time,
                                   desired_start_step_size,
                                   end_time)}
   , previous_time{start_time}
   , start_step_size{next_time - start_time}
   , step_number{0}
 {}



 void
 DiscreteTime::set_desired_next_step_size(const double next_step_size)
 {
   next_time = calculate_next_time(current_time, next_step_size, end_time);
 }



 void
 DiscreteTime::advance_time()
 {
   Assert(next_time > current_time,
          dealii::ExcMessage(
            "You can't advance time further."
            "Either dt == 0 or you are at the end of the simulation time."));
   const double step_size = get_next_step_size();
   previous_time          = current_time;
   current_time           = next_time;
   ++step_number;
   next_time = calculate_next_time(current_time, step_size, end_time);
 }



 void
 DiscreteTime::restart()
 {
   previous_time = start_time;
   current_time  = start_time;
   next_time     = calculate_next_time(current_time, start_step_size, end_time);
   step_number   = 0;
 }



 void
 DiscreteTime::set_end_time(const double new_end_time)
 {
   end_time = new_end_time;

   if (step_number == 0)
     next_time = calculate_next_time(start_time,
                                     start_step_size,
                                     end_time);
   else
     next_time = calculate_next_time(current_time,
                                     get_previous_step_size(),
                                     end_time);
 }

} // namespace RMHD

} // namespace TimeDiscretization