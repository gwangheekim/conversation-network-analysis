#include <iostream>
#include <cstdio>
#include <chrono>

void progressbar(int step, int total)
{
  // progress width
  const int pwidth = 72;
  
  // minus label len
  int pos = (step * pwidth) / total;
  int percent = (step * 100) / total;
  
  // calculate elapsed time in seconds
  //auto current_time = std::chrono::steady_clock::now();
  auto current_time = std::chrono::system_clock::now();
  static auto start_time = current_time;  // static to keep the initial value
  if (step == 1) {
    start_time = current_time;
  }
  
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
  
  // calculate remaining time in seconds
  auto remaining_seconds = ((total - step) * elapsed_seconds) / step;
  
  // Calculate remaining time in hours, minutes, and seconds
  int hours = remaining_seconds / 3600;
  int minutes = (remaining_seconds % 3600) / 60;
  int seconds = remaining_seconds % 60;
  
  // Format start time as HH:MM:SS
  std::time_t start_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm* start_time_tm = std::localtime(&start_time_t);
  char start_time_str[9];
  std::strftime(start_time_str, sizeof(start_time_str), "%H:%M:%S", start_time_tm);
  
  // fill progress bar with =
  std::cout << "[";
  for (int i = 0; i < pos; ++i) {
    std::printf("%c", '=');
  }
  
  // fill progress bar with spaces
  std::printf("%*c", pwidth - pos + 1, ']');
  
  // print percentage, ETA, and total elapsed time
  std::printf(" %3d%% ETA: %02d:%02d:%02d \r", percent, hours, minutes, seconds);
  
  // flush output to make sure it's displayed immediately
  std::cout.flush();
}
