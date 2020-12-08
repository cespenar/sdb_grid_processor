import glob
import itertools
import os
import re

import mesa_reader as mesa
import numpy as np
import pandas as pd
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Model(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    m_i = Column(Float, nullable=False)
    m_env = Column(Float, nullable=False)
    rot_i = Column(Float, nullable=False)
    z_i = Column(Float, nullable=False)
    y_i = Column(Float, nullable=False)
    fh = Column(Float, nullable=False)
    fhe = Column(Float, nullable=False)
    fsh = Column(Float, nullable=False)
    mlt = Column(Float, nullable=False)
    sc = Column(Float, nullable=False)
    reimers = Column(Float, nullable=False)
    blocker = Column(Float, nullable=False)
    turbulence = Column(Float, nullable=False)
    m = Column(Float, nullable=False)
    rot = Column(Float, nullable=False)
    model_number = Column(Integer, nullable=False)
    level = Column(Integer, nullable=False)
    m_he_core = Column(Float, nullable=False)
    log_Teff = Column(Float, nullable=False)
    log_g = Column(Float, nullable=False)
    log_L = Column(Float, nullable=False)
    radius = Column(Float, nullable=False)
    age = Column(Float, nullable=False)
    z_surf = Column(Float, nullable=False)
    y_surf = Column(Float, nullable=False)
    center_he4 = Column(Float, nullable=False)
    custom_profile = Column(Float, nullable=False)
    top_dir = Column(String, nullable=False)
    log_dir = Column(String, nullable=False)


class GridProcessor():
    """Structure containing a raw MESA grid of sdB stars.

    Reads a grid and saves the models into a database.

    Parameters
    ----------
    grid_dir : str
        Directory containing the grid of models.
    output_file : str
        The name of output file.
    output_dir : str
        Output directory for .mod files of selected progenitors.

    Attributes
    ----------
    grid_dir : str
        Directory containing the grid of models.
    db_file : str
        Output database. 

    Examples
    ----------
    >>> grid_dir = "test_grid"
    >>> db_file = "test.db"

    >>> g = GridProcessor(grid_dir, db_file)
    >>> g.evaluate_sdb_grid()

    Here `grid_dir` is a directory containing the calcualted MESA 
    models, and `test.db` is an output database. The grid is at
    first initialized and then it is evaluated and commited to
    the databse.

    """

    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'

    def __init__(self, grid_dir, db_file):
        self.grid_dir = grid_dir
        self.db_file = db_file
        self.engine = create_engine(f'sqlite:///{self.db_file}')
        Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.session = Session()
        self.log_dirs = self.find_all_log_dirs()

    def evaluate_sdb_grid(self):
        """Reads models in a directory tree and creates a grid of parameters.
        Saves the grid to database.

        Parameters
        ----------

        Returns
        ----------
        """

        i = 0
        for log_dir in sorted(self.log_dirs):
            print(log_dir)
            initial_parameters = self.read_logdir_name(
                os.path.basename(log_dir))
            history = mesa.MesaData(os.path.join(log_dir, 'history.data'))
            for profile in sorted(glob.glob(os.path.join(log_dir, 'custom_He*.data')), reverse=True):
                data = mesa.MesaData(profile)
                rx = re.compile(self.numeric_const_pattern, re.VERBOSE)
                custom_profile = rx.findall(os.path.basename(profile))[0]
                if not os.path.isfile(os.path.join(log_dir, f'custom_He{custom_profile}_summary.txt')):
                    print("No pulsational calculation availiable. Skipping the model.")
                    continue
                top_dir = os.path.basename(os.path.split(log_dir)[0])
                self.commit_one_model(i, initial_parameters, history, data.model_number,
                                      custom_profile, top_dir, os.path.basename(log_dir))
                i += 1
            print()

    def find_all_log_dirs(self):
        """Returns list of LOG directories in the grid directory.

        Returns
        ----------
        list
            Flattened list of LOG directories.
        """
        top_dirs = glob.glob(os.path.join(self.grid_dir, 'logs_*'))
        log_dirs = [glob.glob(os.path.join(directory, 'logs_*'))
                    for directory in top_dirs]
        return list(itertools.chain(*log_dirs))

    def commit_one_model(self, i, initial_parameters, history, model_number,
                         custom_profile, top_dir, log_dir):
        """Creates and commits a single row of models table.

        Parameters
        ----------
        i : int
            Index of the row.
        initial_parameters : dict
            Initial parameters of progenitor in dict format.
        history : MesaData
            Evolutionary track (MESA history file) as MesaData object.
        model_numrer : int
            Model number of a selected model.
        custom profile : float
            Approximated central helium abundance reported by profile name. 

        Returns
        ----------
        """

        model = Model(
            m_i=initial_parameters['m_i'],
            m_env=initial_parameters['m_env'],
            rot_i=initial_parameters['rot'],
            z_i=initial_parameters['z'],
            y_i=initial_parameters['y'],
            fh=initial_parameters['fh'],
            fhe=initial_parameters['fhe'],
            fsh=initial_parameters['fsh'],
            mlt=initial_parameters['mlt'],
            sc=initial_parameters['sc'],
            reimers=initial_parameters['reimers'],
            blocker=initial_parameters['blocker'],
            turbulence=initial_parameters['turbulence'],
            m=history.star_mass[model_number-1],
            rot=history.v_surf_km_s[model_number-1],
            model_number=model_number,
            level=initial_parameters['level'],
            m_he_core=history.star_mass[0] - initial_parameters['m_env'],
            log_Teff=history.log_Teff[model_number-1],
            log_g=history.log_g[model_number-1],
            log_L=history.log_L[model_number-1],
            radius=history.radius[model_number-1],
            age=history.star_age[model_number-1],
            z_surf=10.0**history.log_surf_cell_z[model_number-1],
            y_surf=history.surface_he3[model_number-1] +
            history.surface_he4[model_number-1],
            center_he4=history.center_he4[model_number-1],
            custom_profile=custom_profile,
            top_dir=top_dir,
            log_dir=log_dir
        )

        if not self.model_in_database(model):
            self.session.add(model)
            self.session.commit()

    def model_in_database(self, model):
        """Checks if a model is already in the database.

        Parameters
        ----------
        model : Model
            A model to be checked.

        Returns
        ----------
        bool
            Is the model in the database?
        """

        count = self.session.query(Model).filter(
            Model.m_i == model.m_i,
            Model.m_env == model.m_env,
            Model.rot_i == model.rot_i,
            Model.z_i == model.z_i,
            Model.y_i == model.y_i,
            Model.fh == model.fh,
            Model.fhe == model.fhe,
            Model.fsh == model.fsh,
            Model.mlt == model.mlt,
            Model.sc == model.sc,
            Model.reimers == model.reimers,
            Model.blocker == model.blocker,
            Model.turbulence == model.turbulence,
            Model.m == model.m,
            Model.rot == model.rot,
            Model.model_number == model.model_number,
            Model.level == model.level,
            Model.m_he_core == model.m_he_core,
            Model.log_Teff == model.log_Teff,
            Model.log_g == model.log_g,
            Model.log_L == model.log_L,
            Model.radius == model.radius,
            Model.age == model.age,
            Model.z_surf == model.z_surf,
            Model.y_surf == model.y_surf,
            Model.center_he4 == model.center_he4,
            Model.custom_profile == model.custom_profile,
            Model.top_dir == model.top_dir,
            Model.log_dir == model.log_dir
        ).count()

        if count:
            return True
        else:
            return False

    @staticmethod
    def read_logdir_name(log_dir):
        """Recovers initial values from the name of log direcotry.

        Parameters
        ----------
        f_name : str
            Name of log directory.

        Returns
        ----------
        values : dict
            Initial parameters in dict format.
        """

        s = log_dir.split('_')

        values = {}
        values['m_i'] = float(s[1][2:])
        values['m_env'] = float(s[2][4:])
        values['rot'] = float(s[3][3:])
        values['z'] = float(s[4][1:])
        values['y'] = float(s[5][1:])
        values['fh'] = float(s[6][2:])
        values['fhe'] = float(s[7][3:])
        values['fsh'] = float(s[8][3:])
        values['mlt'] = float(s[9][3:])
        values['sc'] = float(s[10][2:])
        values['reimers'] = float(s[11][7:])
        values['blocker'] = float(s[12][7:])
        values['turbulence'] = float(s[13][10:])
        values['level'] = int(s[14][3:])
        values['model_number'] = int(s[15])

        return values
