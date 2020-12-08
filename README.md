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

A directory with containing MESA models.

    grid_dir = "test_grid"

Database containing the final grid:

    db_file = "test.db"

Initialize a grid:

    g = GridProcessor(grid_dir, db_file)

Evaluate the grid. This method evaluated the grid and saves the output to a database.

    g.evaluate_sdb_grid()
