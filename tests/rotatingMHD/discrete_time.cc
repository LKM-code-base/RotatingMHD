#include <deal.II/base/mpi.h>

#include <rotatingMHD/discrete_time.h>

#include <iostream>

namespace RMHD
{

namespace TimeDiscretization
{

void discrete_time_test()
{
  DiscreteTime timestepping(0.0, 1.0);
  std::cout << timestepping << std::endl;
  while (!timestepping.is_at_end())
  {
    timestepping.set_desired_next_step_size(0.1);
    timestepping.advance_time();
    std::cout << timestepping << std::endl;
  }
  return;
}

}  // namespace TimeDiscretization

}  // namespace RMHD

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    dealii::deallog.depth_console(0);

    RMHD::TimeDiscretization::discrete_time_test();


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
