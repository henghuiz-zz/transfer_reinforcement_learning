import numpy as np


def test_performance(game_env, input_tensor, logit_tensor, session,
                     shown=False):
    done = False
    all_reward = 0
    while not done:
        state = game_env.get_image()
        if shown:
            game_env.env.render()
        policy_ins = session.run(
            logit_tensor, feed_dict={input_tensor: np.expand_dims(state, 0)})
        action_ins = np.argmax(policy_ins[0])
        reward, done = game_env.do_action(action_ins)
        all_reward += reward
    return all_reward
