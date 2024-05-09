import numpy as np
import os
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two probability distributions.
    
    Args:
    p (list or np.array): Probability distribution p(s).
    q (list or np.array): Probability distribution q(s).

    Returns:
    float: The KL divergence from p to q.
    """
    # Convert lists to numpy arrays for easier handling
    p = np.array(p)
    q = np.array(q)
    
    # Ensure the probabilities do not contain zeros to avoid division by zero in log
    # Use a small epsilon where q is zero
    epsilon = 1e-10
    q = np.where(q == 0, epsilon, q)

    # Compute the KL divergence
    kl_div = np.sum(p * np.log(p / q), where=(p != 0))  # Only consider terms where p > 0

    return kl_div


def action_disagreement(policy1_actions, policy2_actions):
    assert len(policy2_actions) == len(policy1_actions), "Policy actions must have the same length"
    disagreements = (policy1_actions != policy2_actions).sum()
    total_states = len(policy1_actions)
    return disagreements / total_states


def analyze_policy_disagreement(policy1_actions, policy2_actions):
    """
    Analyze the disagreement between two policies.

    Args:
    policy1_actions (np.array): Actions taken by policy 1.
    policy2_actions (np.array): Actions taken by policy 2.
    """

    # Compute the disagreement between the two policies
    disagreement = action_disagreement(policy1_actions, policy2_actions)
    print(f"Disagreement between the two policies: {disagreement:.2f}")

    # print percentage of each action for policy1 and policy2
    policy1_action_counts = np.bincount(policy1_actions, minlength=6)
    policy2_action_counts = np.bincount(policy2_actions, minlength=6)
    total_states = len(policy1_actions)
    # print("Policy 1 action distribution:")
    # for action, count in enumerate(policy1_action_counts):
    #     print(f"Action {action}: {count / total_states:.2f}")
    # print("Policy 2 action distribution:")
    # for action, count in enumerate(policy2_action_counts):
    #     print(f"Action {action}: {count / total_states:.2f}")
    # # compute delta for each action as percentage
    action_deltas = policy1_action_counts - policy2_action_counts
    action_deltas = action_deltas / total_states
    print("Action deltas:")
    for action, delta in enumerate(action_deltas):
        print(f"Action {action}: {delta:.2f}")



    # # analyze which actions are different
    # different_actions_mask = policy1_actions != policy2_actions

    # # group by different actions
    # different_actions1 = policy1_actions[different_actions_mask]
    # different_actions2 = policy2_actions[different_actions_mask]

    # # count the number of times each action is different
    # action_diff_counts = {}
    # for action1, action2 in zip(different_actions1, different_actions2):
    #     action_pair = (action1, action2)
    #     action_diff_counts[action_pair] = action_diff_counts.get(action_pair, 0) + 1
    
    # #plot the action differences
    # print("Action differences:")
    # for action_pair, count in action_diff_counts.items():
    #     print(f"Action pair: {action_pair}, Count: {count}")

    # # plot the action differences
    # plt.hist2d(different_actions1, different_actions2, bins=(6, 6), range=((-0.5, 5.5), (-0.5, 5.5)))
    # plt.xlabel("Policy 1 Actions")
    # plt.ylabel("Policy 2 Actions")
    # plt.title("Action Differences")
    # plt.colorbar()
    # plt.savefig("action_differences.png")
    # # map action id to action name
    # unpruned_actions = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    # #pruned_actions = ["NOOP", "FIRE", "RIGHT", "LEFT"]

    # regard actions 0,1,4,5 as the same action: "NOOP"
    policy1_actions = np.where((policy1_actions == 4) | (policy1_actions == 5) | (policy1_actions == 1), 0 , policy1_actions)
    policy2_actions = np.where((policy2_actions == 4) | (policy2_actions == 5) | (policy2_actions == 1), 0 , policy2_actions)
    # both policies should now only have actions 0,2,3

    # Compute the disagreement between the two policies
    disagreement = action_disagreement(policy1_actions, policy2_actions)
    print(f"Disagreement between the two policies after mapping actions: {disagreement:.2f}")


    



    
# # Example usage
# p = [0.2, 0.5, 0.3]  # Example probability distribution p(s)
# q = [0.1, 0.4, 0.5]  # Example probability distribution q(s)

# kl_result = kl_divergence(p, q)
# print("KL Divergence: ", kl_result)

from eclaire_utils.utils import get_eclaire_config
eclaire_cfg = get_eclaire_config()

# Example usage
eclaire_dir = eclaire_cfg.eclaire_dir
actions_eclaire = np.load(os.path.join(eclaire_dir, "actions_eclaire.npy"))
best_policy_actions = np.load(os.path.join(eclaire_dir, "best_policy_actions.npy"))

disagreement = action_disagreement(best_policy_actions, actions_eclaire)
print(disagreement)

analyze_policy_disagreement(best_policy_actions, actions_eclaire)