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

Base = declarative_base()  # SQLAlchemy base class


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

    """

    considered_models = np.arange(0.9, 0.05, -0.05)
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'

    def __init__(self, grid_dir, db_file, output_file):
        self.grid_dir = grid_dir
        self.output_file = output_file

        self.db_file = db_file
        self.engine = create_engine(f'sqlite:///{self.db_file}')
        Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.session = Session()

        self.log_dirs = self.find_all_log_dirs()
        self.number_of_models = len(
            self.log_dirs) * len(self.considered_models)
        self.grid = np.zeros(self.number_of_models, dtype=[('m_i', 'float64'),
                                                           ('m_env', 'float64'),
                                                           ('rot_i', 'float64'),
                                                           ('z_i', 'float64'),
                                                           ('y_i', 'float64'),
                                                           ('fh', 'float64'),
                                                           ('fhe', 'float64'),
                                                           ('fsh', 'float64'),
                                                           ('mlt', 'float64'),
                                                           ('sc', 'float64'),
                                                           ('reimers', 'float64'),
                                                           ('blocker', 'float64'),
                                                           ('turbulence',
                                                            'float64'),
                                                           ('m', 'float64'),
                                                           ('rot', 'float64'),
                                                           ('model_number', 'int'),
                                                           ('level', 'int'),
                                                           ('m_he_core',
                                                            'float64'),
                                                           ('log_Teff', 'float64'),
                                                           ('log_g', 'float64'),
                                                           ('log_L', 'float64'),
                                                           ('radius', 'float64'),
                                                           ('age', 'float64'),
                                                           ('z_surf', 'float64'),
                                                           ('y_surf', 'float64'),
                                                           ('center_he4',
                                                            'float64'),
                                                           ('custom_profile',
                                                            'float64'),
                                                           ('top_dir',
                                                            'U100'),
                                                           ('log_dir',
                                                            'U200'),
                                                           ])

    def evaluate_sdb_grid(self):
        """Reads models in a directory tree and creates a grid of parameters.
        Saves the grid to a file.

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
                self.add_one_row(i, initial_parameters, history, data.model_number,
                                 custom_profile, top_dir, os.path.basename(log_dir))
                self.commit_model(i, initial_parameters, history, data.model_number,
                                  custom_profile, top_dir, os.path.basename(log_dir))
                i += 1
            print()

        self.save_grid_to_file()

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

    def add_one_row(self, i, initial_parameters, history, model_number,
                    custom_profile, top_dir, log_dir):
        """Populates a single row of the grid.

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

        self.grid['m_i'][i] = initial_parameters['m_i']
        self.grid['m_env'][i] = initial_parameters['m_env']
        self.grid['rot_i'][i] = initial_parameters['rot']
        self.grid['z_i'][i] = initial_parameters['z']
        self.grid['y_i'][i] = initial_parameters['y']
        self.grid['fh'][i] = initial_parameters['fh']
        self.grid['fhe'][i] = initial_parameters['fhe']
        self.grid['fsh'][i] = initial_parameters['fsh']
        self.grid['mlt'][i] = initial_parameters['mlt']
        self.grid['sc'][i] = initial_parameters['sc']
        self.grid['reimers'][i] = initial_parameters['reimers']
        self.grid['blocker'][i] = initial_parameters['blocker']
        self.grid['turbulence'][i] = initial_parameters['turbulence']
        self.grid['m'][i] = history.star_mass[model_number-1]
        self.grid['rot'][i] = history.v_surf_km_s[model_number-1]
        self.grid['model_number'][i] = model_number
        self.grid['level'][i] = initial_parameters['level']
        self.grid['m_he_core'][i] = history.star_mass[0] - \
            initial_parameters['m_env']
        self.grid['log_Teff'][i] = history.log_Teff[model_number-1]
        self.grid['log_g'][i] = history.log_g[model_number-1]
        self.grid['log_L'][i] = history.log_L[model_number-1]
        self.grid['radius'][i] = history.radius[model_number-1]
        self.grid['age'][i] = history.star_age[model_number-1]
        self.grid['z_surf'][i] = 10.0**history.log_surf_cell_z[model_number-1]
        self.grid['y_surf'][i] = history.surface_he3[model_number -
                                                     1] + history.surface_he4[model_number-1]
        self.grid['center_he4'][i] = history.center_he4[model_number-1]
        self.grid['custom_profile'][i] = custom_profile
        self.grid['top_dir'][i] = top_dir
        self.grid['log_dir'][i] = log_dir

    def commit_model(self, i, initial_parameters, history, model_number,
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

        self.session.add(model)
        self.session.commit()

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

    def save_grid_to_file(self):
        """Saves the processed grid to a text file.

        Parameters
        ----------

        Returns
        ----------
        None
        """

        df = pd.DataFrame(self.grid)
        df.to_csv(self.output_file, header=True, index=False, sep=' ')

    @classmethod
    def set_considered_models(cls, start: float = 0.9, end: float = 0.05, step: float = -0.05) -> np.ndarray:
        """Sets a number of models with positive offset versus the default model.

        Parameters
        ----------
        start : float
            Start of range. Default: 0.9
        end : float
            Start of range. Default: 0.9
        step : float
            Start of range. Default: 0.9

        Returns
        ----------
        """

        cls.considered_models = np.arange(start, end, step)


if __name__ == "__main__":

    grid_dir = "/Users/cespenar/sdb/sdb_grid_processor/test_grid"
    db_file = "test.db"
    output_file = "grid_mi1.0_z0.015_lvl0.txt"

    g = GridProcessor(grid_dir, db_file, output_file)
    log_dirs = g.find_all_log_dirs()
