import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin
from stable_baselines3.common.utils import is_vectorized_box_observation


class Policy(ABC):
    """
    Abstract base class for a policy.

    Parameters
    ----------
    observation_space : gym.Space
        The observation space of the environment.
    action_space : gym.Space
        The action space of the environment.

    Attributes
    ----------
    observation_space : gym.Space
        The observation space of the environment.
    action_space : gym.Space
        The action space of the environment.
    """

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def predict(self, obs, state=None, deterministic=True, episode_start=0):
        """
        Predict the action to take given an observation.

        Parameters
        ----------
        obs : np.ndarray
            The observation input.
        state : object, optional
            The state of the policy (default is None).
        deterministic : bool, optional
            Whether to use a deterministic policy (default is True).
        episode_start : int, optional
            The episode start index (default is 0).

        Returns
        -------
        action : np.ndarray
            The action to take.
        state : object
            The updated state of the policy.
        """
        raise NotImplementedError


class SB3Policy(Policy):
    def __init__(self, base_policy):
        self.base_policy = base_policy
        super().__init__(
            self.base_policy.observation_space, self.base_policy.action_space
        )

    def predict(self, obs, state=None, deterministic=True, episode_start=0):
        return self.base_policy.predict(obs, state, deterministic, episode_start)


class DTPolicy(Policy):
    """
    Decision Tree Policy class.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        The decision tree classifier or regressor.
    env : gym.Env
        The environment in which the policy operates.

    Attributes
    ----------
    clf : sklearn.base.BaseEstimator
        The decision tree classifier or regressor.
    observation_space : gym.Space
        The observation space of the environment.
    action_space : gym.Space
        The action space of the environment.
    """

    def __init__(self, clf, env):
        assert isinstance(env.observation_space, gym.spaces.Box)
        if isinstance(env.action_space, gym.spaces.Box):
            assert isinstance(clf, RegressorMixin)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            assert isinstance(clf, ClassifierMixin)
        super().__init__(env.observation_space, env.action_space)
        self.clf = clf
        # Policy initialization with random samples
        self.clf.fit(
            [self.observation_space.sample() for _ in range(1000)],
            [self.action_space.sample() for _ in range(1000)],
        )

    def predict(self, obs, state=None, deterministic=True, episode_start=0):
        """
        Predict the action to take given an observation.

        Parameters
        ----------
        obs : np.ndarray
            The observation input.
        state : object, optional
            The state of the policy (default is None).
        deterministic : bool, optional
            Whether to use a deterministic policy (default is True).
        episode_start : int, optional
            The episode start index (default is 0).

        Returns
        -------
        action : np.ndarray
            The action to take.
        state : object
            The updated state of the policy.
        """
        if not is_vectorized_box_observation(obs, self.observation_space):
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.clf.predict(obs.reshape(1, -1)).squeeze().astype(int)
            else:
                if self.action_space.shape[0] > 1:
                    action = self.clf.predict(obs.reshape(1, -1)).squeeze()
                else:
                    action = self.clf.predict(obs.reshape(1, -1))
            return action, state
        else:
            if isinstance(self.action_space, gym.spaces.Discrete):
                return self.clf.predict(obs).astype(int), None
            else:
                if self.action_space.shape[0] > 1:
                    return self.clf.predict(obs), None
                else:
                    return self.clf.predict(obs)[:, np.newaxis], None

    def fit(self, S, A):
        """
        Fit the decision tree with the provided observations and actions.

        Parameters
        ----------
        S : np.ndarray
            The observations.
        A : np.ndarray
            The actions.
        """
        self.clf.fit(S, A)


class ObliqueDTPolicy(Policy):
    """
    Oblique Decision Tree Policy class.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        The decision tree classifier or regressor.
    env : gym.Env
        The environment in which the policy operates.

    Attributes
    ----------
    clf : sklearn.base.BaseEstimator
        The decision tree classifier or regressor.
    observation_space : gym.Space
        The observation space of the environment.
    action_space : gym.Space
        The action space of the environment.
    """

    def __init__(self, clf, env):
        if isinstance(env.action_space, gym.spaces.Box):
            assert isinstance(clf, RegressorMixin)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            assert isinstance(clf, ClassifierMixin)
        super().__init__(env.observation_space, env.action_space)
        self.clf = clf
        # Policy initialization with clipped random samples
        init_S = np.array([self.observation_space.sample() for _ in range(1000)]).clip(
            -2, 2
        )
        self.clf.fit(
            self.get_oblique_data(init_S),
            [self.action_space.sample() for _ in range(1000)],
        )

    def get_oblique_data(self, S):
        """
        Generate oblique data by creating pairwise differences between observations.

        Parameters
        ----------
        S : np.ndarray
            The input observations. Each row represents an observation.

        Returns
        -------
        final : np.ndarray
            The original observations stacked with pairwise differences.
        """
        # Generate indices for the lower triangular part of the matrix
        indices = np.tril_indices(S.shape[1], k=-1)
        # Tile the rows to create matrices for subtraction
        a_mat = np.tile(S[:, np.newaxis, :], (1, S.shape[1], 1))
        b_mat = np.transpose(a_mat, axes=(0, 2, 1))

        # Compute the differences and store them in the appropriate location in the result array
        diffs = a_mat - b_mat
        result = diffs[:, indices[0], indices[1]]

        # Stack the original rows with the differences
        final = np.hstack((S, result))
        return final

    def predict(self, obs, state=None, deterministic=True, episode_start=0):
        """
        Predict the action to take given an observation.

        Parameters
        ----------
        obs : np.ndarray
            The observation input.
        state : object, optional
            The state of the policy (default is None).
        deterministic : bool, optional
            Whether to use a deterministic policy (default is True).
        episode_start : int, optional
            The episode start index (default is 0).

        Returns
        -------
        action : np.ndarray
            The action to take.
        state : object
            The updated state of the policy.
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.clf.predict(self.get_oblique_data(obs)).astype(int), None
        else:
            if self.action_space.shape[0] > 1:
                return self.clf.predict(self.get_oblique_data(obs)), None
            else:
                return (
                    self.clf.predict(self.get_oblique_data(obs))[:, np.newaxis],
                    None,
                )

    def fit(self, S, A):
        """
        Fit the decision tree with the provided oblique observations and actions.

        Parameters
        ----------
        S : np.ndarray
            The observations.
        A : np.ndarray
            The actions.
        """
        self.clf.fit(self.get_oblique_data(S), A)
