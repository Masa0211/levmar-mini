# levmar-mini

levmar-mini is a simplified version of the original [levmar](http://users.ics.forth.gr/~lourakis/levmar/) library (Levenberg-Marquardt non-linear least squares optimization) with a modern C++ interface. This library focuses on providing core optimization functionality with no external dependencies beyond the C++ standard library.

## Features

- Implementation of the Levenberg-Marquardt algorithm for non-linear least squares optimization
- Modern C++ interface:
  - Uses function objects/lambdas instead of C-style function pointers
  - Provides structured options and info via C++ classes/structs
  - Uses standard C++ containers instead of raw memory management
- Double precision only (removed single precision to simplify codebase)
- No external dependencies (all LAPACK-dependent functions removed)
- Built-in LU decomposition for matrix operations instead of relying on external libraries

## What's Included

This minimalist version retains only:
- `dlevmar_dif` function (double precision, using finite differences for Jacobian approximation)
- Supporting utility functions required for the core algorithm
- Modern C++ interface around the original algorithm

## What's Removed

- All LAPACK-dependent functions
- Single precision implementation (retained only double precision)
- Various unnecessary utility functions
- Complex C macros for managing different precision types

## Requirements

- C++11 or later
- CMake build system

## Installation

```
# Clone the repository
git clone https://github.com/Masa0211/levmar-mini.git
cd levmar-mini

# Build with CMake
mkdir build
cd build
cmake ..
make
```

## Usage

Include the necessary headers in your project:
```
#include "levmar.h"
```
The main class to use is levmar::LevMar, which provides the Levenberg-Marquardt optimization algorithm.


### Key Features of the Interface

- Use of `std::function` for passing optimization functions
- Structured options via the `Options` struct
- Information returned in an `Info` struct
- Exception safety for invalid inputs
- No raw memory management required from the user
- Uses `std::vector` internally instead of `malloc`/`free`

## License

This project is licensed under the GPL-2.0 License - see the LICENSE file for details.

Based on the original levmar library by Manolis Lourakis, which is also licensed under GPL-2.0.

## Acknowledgments

- Original levmar library: http://users.ics.forth.gr/~lourakis/levmar/

