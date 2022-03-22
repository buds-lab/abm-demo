import warnings
import os
import logging
import json
import random
import gym
import numpy as np
import pandas as pd
from gym.utils import seeding
from common.utils import load_variable, save_variable
from energy_models import Zone


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter("ignore", ResourceWarning)


class Occupant:
    """
    Occupant object based on available data. The dataframes with metadata,
    phisiological, and environmental data will be loaded and then filtered
    based on the `user_ids`
    """

    def __init__(self, user_id, pcm):
        self.user_id = user_id
        self.pcm = pcm
        self.tp_real = []  # thermal preference ground truth
        self.tp_pred = []  # thermal preference prediction
        self.occupied_time_step = []


def occupant_loader(
    num_new_occupants,
    occupant_tolerance,
    occupant_tol_file,
    occupant_background,
    occupant_preference,
    occupant_pcm,
):
    """
    Create occupant objects based on real dataset and separates them in
    train and test occupants

    Parameters
    ----------
    num_new_occupants: int
        Number of occupants that start the day
    occupant_tolerance: float
        Number in [0,1] that determines how tolerant the occupant its to the
        environment. This number is then used to sample from an occupant
        distribution. A number of -1 means using only real occupants.
    occupant_tol_file: str
        Occupant's tolerance csv path
    occupant_background: str
        Occupants' background csv path
    occupant_preference: str
        Occupants' thermal preference csv path
    occupant_pcm: str
        Occupants' PCM csv path

    Returns
    -------
    dict_occ
        Dictionary with all occupants' objects as values and `user_id` as key.
    df_occ
        Dataframe with historical data of the real occupants.
    """
    dict_occ = {}

    list_all_occ = list(pd.read_csv(occupant_background)["user_id"])
    dict_pcm = load_variable(occupant_pcm)

    # using only real occupants from dataset
    if occupant_tolerance == -1:
        assert len(list_all_occ) >= num_new_occupants
        list_occ = random.sample(list_all_occ, num_new_occupants)

        # initialize new occupant objects
        for occupant in list_occ:
            dict_occ[occupant] = Occupant(occupant, dict_pcm[occupant])

        # historical data for all current occupants
        df_occ = pd.read_csv(occupant_preference[occupant_preference["user_id"] in list_occ])

    # generating synthetic occupants based on real ones from dataset
    else:
        # find and load occupants within the tolerance threshold
        df_occ_tol = pd.read_csv(occupant_tol_file)
        df_filtered_occupants = df_occ_tol[df_occ_tol["tolerance"] <= occupant_tolerance * 100]
        list_real_occ = list(df_filtered_occupants["user_id"])

        # initialize new occupants objects
        for occupant in range(1, num_new_occupants + 1):
            user_id = f"user_{occupant}"
            # TODO: below can be changed so that it's not the same PCM for more than one user
            real_occ = random.sample(list_real_occ, 1)[0]  # use a real occupant's pcm
            dict_occ[user_id] = Occupant(user_id, dict_pcm[real_occ])

        # historical data from real occupants
        df_occ = pd.read_csv(occupant_preference)
        df_occ = df_occ[df_occ["user_id"].isin(list_real_occ)]

    # TODO: load cohort model
    return dict_occ, df_occ


def zone_loader(data_path, zone_attributes, weather_file):
    """
    Load information about the different zones.

    Parameters
    ----------
    data_path: str
        Use case folder path
    zone_attributes: str
        JSON file name with zones' attributes
    weather_file: str
        Weather file name

    Return
    -------
    zones
        Dictonary with all zones' objects with `zone_id` as key.
    observation_spaces
        List of observation space for all zones
    action_spaces
        List of action space for all zones
    """

    with open(zone_attributes) as json_file:
        data = json.load(json_file)

    zones, observation_spaces, action_spaces = {}, [], []

    # Initialize zone objects based on attributes file
    for uid, attributes in zip(data, data.values()):
        # zone object
        zone = Zone(
            zone_id=uid,
            model_type=attributes["model_type"],
            model_features=attributes["model_features"],
            max_num_occupants=attributes["max_num_occupants"],
        )

        # load zone-specific indoor and weather data file
        data_file = str(uid) + ".csv"
        indoor_data = data_path / data_file
        with open(indoor_data) as csv_file:
            data = pd.read_csv(csv_file)

        zone.data["t_set"] = list(data["t_set"])
        zone.data["t_in"] = list(data["t_in_sensor"])
        zone.data["rh_in"] = list(data["rh_in"])
        zone.data["hour"] = list(data["hour"])
        zone.data["day"] = list(data["day_type"])

        with open(weather_file) as csv_file:
            weather_data = pd.read_csv(csv_file)

        zone.data["t_out"] = list(weather_data["t_out"])
        zone.data["rh_out"] = list(weather_data["rh_out"])

        # data-driven model for the zone
        zone.train_model()

        # TODO: lower and upper bound for states and actions. Needed when DRL
        #    is used, Q-networks and so on.

        observation_spaces.append(zone.observation_space)
        action_spaces.append(zone.action_space)

        # zones = {uid: zone object}
        zone.reset()
        zones[uid] = zone

    return zones, observation_spaces, action_spaces


class ComfortLearn(gym.Env):
    def __init__(
        self,
        experiment_name,
        data_path,
        num_new_occupants,
        occupant_timing,
        occupant_tolerance,
        occupant_tol_file,
        occupant_preference,
        occupant_background,
        occupant_pcm,
        zone_attributes,
        weather_file,
        zones_states_actions,
        simulation_period=(0, 23520 - 1),  # every 15min
        cost_function=["unc"],
        central_agent=True,
        verbose=True,
    ):
        self.folder_str = experiment_name
        self.real_occ = True if occupant_tolerance == 1 else False

        # folder for experiment results
        try:
            os.mkdir(self.folder_str)
        except OSError:
            pass

        # logging file
        logging.basicConfig(
            filename=self.folder_str + "/" + self.folder_str + ".log",
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
        )

        # load parameters
        with open(zones_states_actions) as json_file:
            self.zones_states_actions = json.load(json_file)

        # create occupants objects and load their data
        msg = f"Creating occupants based on data from {data_path} ..."

        logging.info(msg)
        if verbose:
            print(msg)

        params_occupant = {
            "num_new_occupants": num_new_occupants,
            "occupant_tolerance": occupant_tolerance,
            "occupant_tol_file": data_path / occupant_tol_file,
            "occupant_background": data_path / occupant_background,
            "occupant_preference": data_path / occupant_preference,
            "occupant_pcm": data_path / occupant_pcm,
        }

        dict_occupants, self.df_occupants = occupant_loader(**params_occupant)

        # generate occupant entering and leaving timings
        self.occ_timing = occupant_timing
        if self.occ_timing == "fixed":
            # enter = 9am, leave = 5pm
            self.enter_time = 9
            self.leave_time = 17
        elif self.occ_timing == "stochastic":
            # randomly sample with standard deviation 2
            # enter = mean of 9am, leaving mean of 5pm
            self.enter_time = np.random.normal(9, 2)
            while self.enter_time <= 7.0:
                # make sure it's above 7am
                self.enter_time = np.random.normal(9, 2)
            self.leave_time = np.random.normal(17, 2)
        else:
            print(f"`occupant_timing` only supports `fixed` or `stochastic` and you type{self.occ_timing})")

        # create zone objects and load their data
        msg = "Creating zones ..."
        logging.info(msg)
        if verbose:
            print(msg)

        params_loader = {
            "data_path": data_path,
            "zone_attributes": data_path / zone_attributes,
            "weather_file": data_path / weather_file,
        }
        (
            self.zones,
            self.observation_spaces,
            self.action_spaces,
        ) = zone_loader(**params_loader)

        self.zones_unc = {}  # unc per zone per occupant, dict of dict
        self.zones_unc_avg = {}  # average unc per zone
        self.simulation_period = simulation_period
        self.cost_function = cost_function
        self.verbose = verbose
        self.n_zones = len(list(self.zones))

        # randomly assigned users to zones
        for occupant_id, occupant in dict_occupants.items():
            curr_zone = "Zone_" + str(random.randint(1, self.n_zones))
            self.zones[curr_zone].occupants[occupant_id] = occupant

        self.reset()

        msg = "Environment created!"
        logging.info(msg)
        if verbose:
            print(msg)

    def get_state_action_spaces(self):
        """Returns state-action spaces for all zones"""
        return self.observation_spaces, self.action_spaces

    def next_time_step(self):
        """Advances simulation to the next time-step"""
        self.time_step = next(self.min_15)
        for zone in self.zones.values():
            zone.time_step = self.time_step

    def step(self, actions):
        s = []  # list of states

        for uid, zone in self.zones.items():
            if self.verbose:
                print(f"Zone: {uid}")
                print(f"Actions to take: {actions}")
                print(f"Current states: {self._get_ob()}")
                print(f"Current occupants: {zone.occupants.keys()}")

            # take actions
            for state_name, value in self.zones_states_actions[uid]["states"].items():
                if actions is None:
                    # no actions are taken, just go through operational data
                    if value:
                        s.append(zone.data[state_name][self.time_step])

                else:
                    pass  # TODO: actually take actions from controller

            # calculate new states TODO

            # sampling enter and leave time for every new day
            if (
                self.occ_timing == "stochastic"
                and self.curr_day != self.next_day
            ):
                print("new day!")
                # randomly sample with standard deviation 2
                # enter = mean of 9am, leaving mean of 5pm
                self.enter_time = np.random.normal(9, 2)
                while self.enter_time <= 7.0:
                    # make sure it's above 7am
                    self.enter_time = np.random.normal(9, 2)
                self.leave_time = np.random.normal(17, 2)

            # calculate cost
            if (
                zone.data["hour"][self.time_step] >= self.enter_time
                and zone.data["hour"][self.time_step] <= self.leave_time
                and zone.data["day"][self.time_step] not in [5, 6]
            ):
                print("timings")
                print(self.enter_time)
                print(self.leave_time)
                for occupant_id, occupant in zone.occupants.items():

                    if self.real_occ:
                        df_curr_occupant = self.df_occupants[
                            self.df_occupants["user_id"] == occupant_id
                        ]
                    else:
                        # using all occupants below the threshold
                        df_curr_occupant = self.df_occupants.copy()

                    # ground truth label, find the closest historical tp votes
                    #   and randomly choose one
                    df_curr_zone_data = pd.DataFrame.from_dict(zone.data)
                    df_curr_occupant["t_in_delta"] = (
                        df_curr_occupant["t-ubi"] - df_curr_zone_data["t_in"]
                    ).abs()
                    df_curr_occupant = df_curr_occupant.sort_values(["t_in_delta"])[:5]
                    tp_gt = df_curr_occupant.sample()["thermal_cozie"].values[0]
                    occupant.tp_pred.append(tp_gt)

                self.zones_unc, self.zones_unc_avg = zone.unc()

            # calculate reward
            rewards = 0  # TODO: placeholder
            self.cumulated_reward_episode += rewards
            self.state = np.array(s)  # states are appended just as a list

        print(f"New states: {self._get_ob()}")

        # update current day, advance time step, and get next day
        self.curr_day = self.next_day
        self.next_time_step()
        for uid, zone in self.zones.items():
            # only one zone is needed for this update
            if self.time_step < len(zone.data["day"]):  # don't overflow
                self.next_day = zone.data["day"][self.time_step]

        return self._get_ob(), rewards, self._terminal()

    def reset(self):
        """Variables initialization"""
        self.min_15 = iter(
            np.array(range(self.simulation_period[0], self.simulation_period[1] + 1))
        )
        self.next_time_step()

        self.cumulated_reward_episode = 0
        self.zones_unc = {}
        self.zones_unc_avg = {}
        s = []

        for zone_id, zone in self.zones.items():
            zone.reset()
            for state_name, value in self.zones_states_actions[zone_id][
                "states"
            ].items():
                if value:
                    s.append(zone.data[state_name][self.time_step])

            # placeholder initialization
            self.curr_day = zone.data["day"][self.time_step]
            self.next_day = zone.data["day"][self.time_step]

        self.state = np.array(s)

        return self._get_ob()

    def _get_ob(self):
        return self.state

    def _terminal(self):
        is_terminal = bool(self.time_step >= self.simulation_period[1])
        if is_terminal:
            for zone in self.zones.values():
                zone.terminate()

            # When the simulation is over, convert all the control variables to
            # numpy arrays so they are easier to plot

            # TODO transform all the control variables to numpy arrays

            # save variables
            for zone_str, zone in self.zones.items():
                save_variable(self.folder_str + "/" + zone_str + ".pkl", zone)

            variables = {
                "dict_zones_unc": self.zones_unc,
                "dict_zones_unc_avg": self.zones_unc_avg,
                "cumulated_reward": self.cumulated_reward_episode,
            }
            for name, var in variables.items():
                save_variable(self.folder_str + "/" + name + ".pkl", var)

            if self.verbose:
                msg = f"Cumulated reward: {str(self.cumulated_reward_episode)}"
                logging.info(msg)
                print(msg)

        return is_terminal

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_zones_unc(self):
        return self.zone_unc

    def cost(self):
        pass  # TODO
