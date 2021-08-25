#include <rotatingMHD/discrete_time.h>

#include <iostream>
#include <sstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

void test()
{
  std::stringstream ss;

  using namespace RMHD::TimeDiscretization;
  {
    DiscreteTime timestepping(0.0, 1.0);
    std::cout << timestepping << std::endl;
    while (!timestepping.is_at_end())
    {
      timestepping.set_desired_next_step_size(0.1);
      timestepping.advance_time();
      std::cout << timestepping << std::endl;
    }

    boost::archive::text_oarchive oa{ss};
    oa << timestepping;
  }
  {
    // deserialize
    boost::archive::text_iarchive ia{ss};
    DiscreteTime  timestepping;
    ia >> timestepping;
    std::cout << timestepping << std::endl;

    timestepping.set_end_time(2.0);
    while (!timestepping.is_at_end())
    {
      timestepping.set_desired_next_step_size(0.1);
      timestepping.advance_time();
      std::cout << timestepping << std::endl;
    }
  }
  return;
}



int main(void)
{
  try
  {
    test();

  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
