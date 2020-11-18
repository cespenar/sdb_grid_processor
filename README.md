# progenitor_grid
=================

Tool for processing a calculated grid of MESA sdB models.

## Installation
Install by cloning the repository, `cd` into it and then execute

    pip install .
    
to  install the package on your system.

## Uninstallation
Uninstall by executing

    pip uninstall sdb_grid_processor

## Sample usage

A directory with top-level directories containing MESA log directories for a given initial mass and metallicity.

    grid_dir = "test_grid"

Output .txt file for the final grid:

    output_file = "test_grid.txt"

Initialize a grid:

    g = GridProcessor(grid_dir, output_file)

Evaluate the grid. This method evaluated the grid and saves the output to a file.

    g.evaluate_initial_grid()
