#pragma once
#include  <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#ifdef USE_NEW_FILESYSTEM_HEADER
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif
#include <set>
#include <cstdlib>
#include <map>
#include <cvsba/cvsba.h>

using namespace std;
#ifdef USE_NEW_FILESYSTEM_HEADER
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif