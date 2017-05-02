import tensorflow as tf
import numpy as np
import os

from transfer_rl.networks import CNNTypeOne, CNNTypeTwo
from transfer_rl.student_learner import StudentLearner
from transfer_rl.environment import GymEnvironment
from transfer_rl.preformance import test_performance


if __name__ == '__main__':
    train_type = 1
    batch_size = 200
    max_iter_number = 2000

    if train_type == 1 or train_type == 3:
        train_games = ['AirRaid-v0', 'Carnival-v0', 'Breakout-v0',
                       'DemonAttack-v0']
        test_game = 'SpaceInvaders-v0'
    else:
        train_games = ['SpaceInvaders-v0', 'Pong-v0']
        test_game = 'Breakout-v0'

    num_game = len(train_games)
    save_path_base = '../save/policy_regression/task' + str(train_type) +'/LfD/'
    if not os.path.isdir(save_path_base):
        os.makedirs(save_path_base)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if train_type == 3:
            student_network = CNNTypeTwo(test_game, 6, True, prefix='student_')
        else:
            student_network = CNNTypeOne(test_game, 6, True, prefix='student_')

        optimizer = tf.train.AdamOptimizer()

        student = StudentLearner(student_network, optimizer)

        sess.run(tf.global_variables_initializer())
        teacher_list = []
        game_env_list = []
        for game_name in train_games:
            # These game all have 6 actions
            game_env = GymEnvironment(game_name)
            teacher_net = CNNTypeOne(game_name, 6, False)
            load_path = '../save/teacher/' + game_name
            checkpoint = tf.train.get_checkpoint_state(load_path)
            teacher_net.load(sess, checkpoint.model_checkpoint_path)

            game_env_list.append(game_env)
            teacher_list.append(teacher_net)

        sum_writer = tf.summary.FileWriter(save_path_base)

        for iter_num in range(max_iter_number):
            now_play = iter_num % num_game
            this_env = game_env_list[now_play]
            this_teacher = teacher_list[now_play]

            state_batch = []
            policy_batch = []
            correct_actions = 0
            for _ in range(batch_size):
                state = this_env.get_image()

                teachers_policy = sess.run(
                    this_teacher.policy,
                    feed_dict={this_teacher.x: np.expand_dims(state, 0)}
                )

                action = np.random.choice(6, 1, p=teachers_policy[0])
                this_env.do_action(action)

                state_batch.append(state)
                policy_batch.append(teachers_policy[0])

            loss, _ = sess.run(
                [student.loss, student.train_step],
                feed_dict={student.input_x: state_batch,
                           student.input_y: policy_batch}
            )

            print(iter_num, loss)

            if (iter_num+1) % 10 == 0:
                student_performance = []
                for id_game in range(num_game):
                    student_score = test_performance(
                        game_env_list[id_game], student.input_x,
                        student_network.logits, sess
                    )

                    student_performance.append(student_score)

                for id_game in range(num_game):
                    record_name = train_games[id_game] + '_score'
                    game_score = student_performance[id_game]
                    sum_writer.add_summary(tf.Summary(value=[
                        tf.Summary.Value(tag=record_name,
                                         simple_value=game_score)]), iter_num)
                sum_writer.flush()

                student_network.save_variable(
                    sess, save_path_base+'checkpoint', iter_num)