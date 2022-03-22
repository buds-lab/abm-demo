from gym import spaces
import numpy as np
from sklearn.linear_model import LinearRegression


class Zone:
    def __init__(
        self,
        zone_id,
        model_type,
        model_features=[],
        max_num_occupants=1,
    ):
        """
        Parameters
        ----------
        zone_id: int
            Zone ID
        model_fetures: list
            List of strings for features to be used for a data-driven model
        model_type: str
            String of the type of thermal model to be used
        max_num_occupants: int
            Maximum number of allowed occupants in the zone
        """

        # Zone attributes
        self.zone_id = zone_id
        self.model_type = model_type
        self.max_num_occupants = max_num_occupants

        self.observation_space = None
        self.action_space = None
        self.time_step = 0
        self.data = {}  # indoor and outdoor data

        self.model_features = model_features
        self.model = None  # data-driven model of the zone
        self.occupants = {}  # {user_id: occupant_object}

    def is_available(self):
        """Check if there is space for more occupants in the zone"""
        if len(self.occupants) >= self.max_num_occupants:
            return False
        return True

    def set_state_space(self, high_state, low_state):
        """Setting the state space and the lower and upper bounds of each state-variable"""
        self.observation_space = spaces.Box(
            low=low_state, high=high_state, dtype=np.float32
        )

    def set_action_space(self, max_action, min_action):
        """Setting the action space and the lower and upper bounds of each action-variable"""
        self.action_space = spaces.Box(
            low=min_action, high=max_action, dtype=np.float32
        )

    def train_model(self):
        """Train a data-driven model based on `model_type`"""
        if self.model_type == "csv":
            # no real model, just operational data
            _model = 0
        elif self.model_type == "reg":
            # simple linear regression
            X = self.data[self.model_features].to_numpy()
            y = self.data["t_in"]
            _model = LinearRegression().fit(X, y)

        elif self.model_type == "lstm":
            # TODO
            _model = None
            print("LSTMs are not yet implemented!")
        else:
            _model = None
            print("Oh no! Your selected model for the zone has not been implemented")

        self.model = _model

        assert self.model is not None

    def get_pmv(self, df_data):
        """
        Calculates the PMV for each current occupant and also the average
        within the zone
        """
        # individual PMV
        dict_pmv = {}
        for user_id, occupant in self.occupants.items():
            # df_occupant =
            dict_pmv[user_id] = 0  # TODO: use simplified_pmv_model
        avg_pmv = sum(dict_pmv.values()) / len(dict_pmv)

        return dict_pmv, avg_pmv

    def unc(self):
        """
        Calculates the Unmet Comfort ratio for each current occupant and also
        the average within the zone. Assumes comfort value is 10.0

        UNC = time with thermal discomfort / occupied time

        Thermal discomfort is anything different than
        thermal preference = "no change"

        UNC in [0, 1] the lower the better.
        """
        # individual UNC
        dict_unc = {}

        for user_id, occupant in self.occupants.items():
            if len(occupant.tp_pred) == 0:
                continue

            dict_unc[user_id] = sum(
                tp != 10.0 for tp in occupant.tp_pred
            ) / len(occupant.tp_pred)

        # average UNC
        avg_unc = sum(dict_unc.values()) / len(dict_unc)

        return dict_unc, avg_unc

    def reset(self):
        pass # TODO

    def terminate(self):
        # TODO: save current status of variables, like UNC

        pass
