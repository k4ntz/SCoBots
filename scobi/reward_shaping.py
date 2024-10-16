from typing import Sequence
import numpy as np
from scobi.utils.game_object import get_wrapper_class

GameObject = get_wrapper_class()


def get_reward_fn(env: str):
    return globals()[f"reward_{env.lower().split('-')[0]}"]


def reward_pong(game_objects: Sequence[GameObject], _: bool) -> float:
    """Small negative reward when y-distance between player and ball > 14"""
    ball, player = _get_game_objects_by_category(game_objects, ["Ball", "Player"])

    if ball is None or player is None:
        return 0

    y_distance = abs(ball.xy[1] - player.xy[1])
    return min(0, 14 - y_distance) * 0.05


last_lives = None
episode_starts = True
last_crashed = False


def reward_seaquest(game_objects: Sequence[GameObject], terminated: bool) -> float:
    """Negative reward on life-loss, positive (scaled) reward on score"""
    global last_lives, episode_starts, last_crashed

    # Identify relevant objects
    player, score, lives, oxygen = _get_game_objects_by_category(game_objects,
                                                                 ["Player", "PlayerScore", "Lives", "OxygenBar"])

    if score is None or episode_starts:
        score_reward = 0
    else:
        score_reward = score.value_diff

    # Discourage crashes
    if player is not None and player.ocgo.crashed and not last_crashed: # new crash detected
        crash_reward = -20
    else:
        crash_reward = 0
    crash_reward = 0

    # Encourage oxygen refill when oxygen is low
    if not last_crashed and oxygen is not None and oxygen.w < 8:
        refill_reward = - player.dy
    else:
        refill_reward = 0
    refill_reward = 0

    last_lives = lives
    episode_starts = terminated
    last_crashed = player.ocgo.crashed

    return score_reward + crash_reward + refill_reward


last_y_distance = None
last_n_lives = 1


def reward_kangaroo(game_objects: Sequence[GameObject], terminated: bool) -> float:
    """Positive Reward for decreasing y-distance to Child Kangaroo."""
    global last_y_distance, last_n_lives, episode_starts, last_crashed

    score, player, child = _get_game_objects_by_category(game_objects, ["Score", "Player", "Child"])
    n_lives = _count_game_objects_of_category(game_objects, "Life")

    # Get current platform
    platform = np.ceil((player.xy[1] - 16) / 48)  # 0: topmost, 3: lowest platform

    # Encourage moving to the child
    if not episode_starts and not last_crashed:
        if platform % 2 == 0:  # even platform, encourage left movement
            movement_reward = - player.dx
        else:  # encourage right movement
            movement_reward = player.dx

        # Always reward upward movement
        movement_reward -= player.dy / 5
    else:
        movement_reward = 0

    if score is not None and not episode_starts:
        # # Discourage monkey kicking (each monkey gives exactly 200 points)
        if score.value_diff == 200:
            score_reward = score.value_diff / 50
        else:
            score_reward = score.value_diff / 5
    else:
        score_reward = 0

    # Discourage loosing lives
    if player is not None and player.ocgo.crashed and not last_crashed: # new crash detected
        crash_reward = -50
    else:
        crash_reward = 0

    last_crashed = player.ocgo.crashed
    episode_starts = terminated

    return score_reward + movement_reward + crash_reward


def reward_skiing(game_objects: Sequence[GameObject], terminated: bool) -> float:  # TODO: update
    # i = 0
    #
    # player_position_idxs = np.empty(0)
    # flag_center_idxs = np.empty(0)
    # flag_velocity_idxs = np.empty(0)
    # for feature in fv_description:
    #     i += 1
    #     feature_name = feature[0]
    #     feature_signature = feature[1]
    #     if feature_name == "CENTER":
    #         input1 = feature_signature[0]
    #         input2 = feature_signature[1]
    #         if input1[0] == "POSITION" and input1[1] == "Flag1" and input2[0] == "POSITION" and input2[
    #             1] == "Flag2":
    #             flag_center_idxs = np.where(fv_backmap == i - 1)[0]
    #     if feature_name == "POSITION":
    #         if feature_signature == "Player1":
    #             player_position_idxs = np.where(fv_backmap == i - 1)[0]
    #     if feature_name == "DIR_VELOCITY":
    #         input = feature_signature[0]
    #         if input[0] == "POSITION_HISTORY" and input[1] == "Flag1":
    #             flag_velocity_idxs = np.where(fv_backmap == i - 1)[0]
    # if not (player_position_idxs.any() and flag_center_idxs.any() and flag_center_idxs.any()):
    #     return None
    #
    # # reward for high player velocity and player decreases euc-distance to center of flag1 and flag2
    # def reward(fv, c_idxs=flag_center_idxs, p_idxs=player_position_idxs, v_idxs=flag_velocity_idxs):
    #     p_entries = fv[p_idxs[0]:p_idxs[-1] + 1]
    #     c_entries = fv[c_idxs[0]:c_idxs[-1] + 1]
    #     v_entries = fv[v_idxs[0]:v_idxs[-1] + 1]
    #     euc_dist = FUNCTIONS["EUCLIDEAN_DISTANCE"]["object"]
    #     player_flag_distance = euc_dist(p_entries, c_entries)[0]
    #     self.reward_history[0] = self.reward_history[1]
    #     self.reward_history[1] = player_flag_distance
    #     delta = self.reward_history[0] - self.reward_history[1]  # decrease in distance: positive sign
    #     player_flag_distance_delta = delta if delta > 0 else 0  # only give positives
    #     euc_velocity_flag = np.clip(math.sqrt((v_entries[0]) ** 2 + (v_entries[1]) ** 2), 0, 10)  # clip to 10
    #     return euc_velocity_flag + 4 * player_flag_distance_delta
    #
    # return reward

    return 0


def _get_game_objects_by_category(game_objects: Sequence[GameObject], categories: Sequence[str]) -> Sequence[GameObject]:
    result = len(categories) * [None]

    for game_object in game_objects:
        for c, category in enumerate(categories):
            if game_object.category == category:
                result[c] = game_object

    return result


def _count_game_objects_of_category(game_objects: Sequence[GameObject], category: str):
    cnt = 0
    for game_object in game_objects:
        if game_object.category == category:
            cnt += 1
    return cnt