import glob
import os
import shutil
from datetime import datetime
from zipfile import ZipFile

import sdb_grid_processor


def backup_db(db_file, backup_dir):
    """Backs up a database.

    Parameters
    ----------
    db_file : str
        Database.
    backup_dir : str
        Destionation directory for backup.

    Returns
    ----------
    """
    if not os.path.isdir(backup_dir):
        os.mkdir(backup_dir)
    now = datetime.now()
    backup_file = f"{db_file}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    shutil.copy(db_file, os.path.join(backup_dir, backup_file))


if __name__ == "__main__":

    grid_dir = "test_grid_dir"
    backup_dir = "test_backup_dir"
    database = "test_sdb_grid.db"

    for f_name in sorted(glob.glob(os.path.join(grid_dir, '*.zip'))):
        if os.path.isfile(database):
            backup_db(database, backup_dir)
        print(f"Processing: {f_name}")
        with ZipFile(f_name, 'r') as archive:
            archive.extractall(path=grid_dir)
            print(f"{f_name} extracted!")
            g = sdb_grid_processor.GridProcessor(grid_dir, database)
            g.evaluate_sdb_grid()
            for top_dir in glob.glob(os.path.join(grid_dir, 'logs_*')):
                if os.path.isdir(top_dir):
                    shutil.rmtree(top_dir)
                    print(f"{top_dir} deleted!")
