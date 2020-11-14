# using turtle for 100 games takes 37.4 seconds (14.1% hit rate using random moves)
# using my own turtle class for 100 games takes 2.4 seconds (we use this for training and testing) (14.2% hit rate using random moves)
import random
import time
import collections
import numpy as np
import pong
from tensorflow.keras import Sequential
from tensorflow.keras.models import *
from tensorflow.keras.layers import Conv2D, Flatten, Dense


# from openai

def rank(x: np.ndarray):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def centered_ranker(x: np.ndarray) -> np.ndarray:
    y = rank(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return np.squeeze(y)


def generate_random_moves_and_save():
    frame_stack, moves, rewards = pong.play_games(10000, 21, single_player=True, show_moves=False, game_agent=None,
                                                  save_moves=True, use_turtle=False)
    print(len(frame_stack))


# generate_random_moves_and_save()
def make_model_play(model_name, games, show_moves, save_moves=False):
    model = load_model(model_name)
    frames, moves, rewards = pong.play_games(games, 10, show_moves=show_moves, save_moves=save_moves, game_agent=model,ball_speed=100, use_turtle=True)
    print(sum(rewards))
    return frames, moves, rewards
#make_model_play("model_saved49", 10, True, False)

# make_model_play("model_saved", 10, False, False)

def train_model_with_feedback_save_n_times(model, n=5):
    model_name = model
    model = load_model(model)
    for x in range(n):
        frames, moves, rewards = make_model_play(model_name, 1000, show_moves=False)
        print("misses {} hits{}".format(sum([x for x in rewards if x == -1]), sum([x for x in rewards if x == 1])))
        trained_model = train_model(model, frames, moves, rewards, 20, 200)
        model.save("model_".format((n), ))
        model_name = "model_" + str(n)
        print("SAVED MODEL" + str(n))


def load_saved_data():
    with open("frame_stack.npy", "rb") as f:
        frame_stack = np.load(f)
    with open("moves.npy", "rb") as f:
        moves = np.load(f)
    with open("rewards.npy", "rb") as f:
        rewards = np.load(f)
    return frame_stack, moves, rewards


def make_a_model(dim_inputs=4, layer_1_nodes=300, layer_2_nodes=50, output_nodes=1):
    model = Sequential()
    model.add(Dense(layer_1_nodes, input_dim=dim_inputs, activation='tanh'))
    model.add(Dense(layer_2_nodes, activation='tanh'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_data_make_train_play():
    frame_stack, moves, rewards = load_saved_data()
    model = make_a_model(4, 300, 25, 1)
    rewards = pong.discount_rewards(rewards, 0.95)
    trained_model = train_model(model, frame_stack, moves, rewards, 5, 100)
    pong.play_games(10, 21, single_player=True, show_moves=True, game_agent=trained_model, save_moves=False)


def train_model(model, frame_stack, moves, rewards, epoch_count, batch_size_specified):
    training_set_moves = []
    training_set_frames = []
    # get rid of moves and frames that won't benefit us
    data = []
    for x in range(len(rewards)):
        if (rewards[x] > 0.5):
            # data.append([frame_stack[x]]+moves[x])
            training_set_frames.append(frame_stack[x])
            training_set_moves.append(moves[x])
    training_set_frames = np.array(training_set_frames, dtype=float)
    training_set_moves = np.array(training_set_moves, dtype=float)
    training_set_moves = training_set_moves.reshape(training_set_moves.shape[0], 1)

    print("Frames dimensions {}".format((len(training_set_frames), training_set_frames[0].shape)))
    # print(training_set_frames[1])
    # print(training_set_frames[2])
    # print(training_set_frames[3])
    # print(training_set_frames[4])
    print("Moves dimensions {}".format(training_set_moves.shape))
    # print(training_set_moves)
    print("Training set has {} items".format(len(training_set_moves)))
    # good_moves=rewards[rewards>0.05]
    # bad_moves=rewards[rewards<-0.05]
    print("model made")
    model.fit(training_set_frames, training_set_moves, epochs=epoch_count, batch_size=batch_size_specified)
    return model


def make_train_and_return_model():
    frame_stack, moves, rewards = pong.play_games(5000, 21, show_moves=False, save_moves=True)
    model = make_a_model(4, 50, 10, 1)
    model = train_model(model, frame_stack, moves, rewards, 10, 200)
    model.save("model_saved")
    # print(frame_stack[1].reshape(1,9))
    # print(model.predict(frame_stack[1].reshape(1,9)))
    pong.play_games(10, 20, game_agent=model)


# make_train_and_return_model()
def load_model_train_on_play_data_return_results(model_name, games_to_play=100):
    model = load_model("model_saved")
    frames, moves, rewards = load_saved_data()
    model = train_model(model, frames, moves, rewards, 10, 3)
    frames, moves, rewards = pong.play_games(10, 10, show_moves=False, save_moves=False,
                                             game_agent=model)
    return frames, moves, rewards


def make_random_model():
    model = Sequential()

    model.add(Dense(64, input_dim=4, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def set_model_weights(model, weights):
    ind = 0
    for layer in model.layers:
        layer_weights = layer.get_weights()[0]
        # print(f"layer shape {layer.get_weights()[0].shape}")
        layer.set_weights(
            [np.reshape(weights[ind:layer_weights.size + ind], layer_weights.shape), layer.get_weights()[1]])
        ind = layer_weights.size
    return model


def get_model_weights(model):
    weights = []

    for layer in model.layers:
        # print(layer.get_weights()[0].shape)
        weights.append(np.ravel(layer.get_weights()[0]))
    return np.concatenate(weights)


def main():
    my_model = make_a_model(3, 64, 32)  # my main model
    theta = get_model_weights(my_model)  # my main weights
    std_dev = 0.02
    lr = 0.1
    generations = 500  # instaed of epoch/episode

    n_trials = 50
    for generation in range(generations):
        print(f"Season {generation}")
        noise = []
        fitness = []

        for n in range(n_trials):
            if(n%20==0):
                print(n)
            noise.append(   np.random.randn(   len(theta) ) * std_dev)
            set_model_weights(my_model, noise[-1] + theta)
            frames, moves, rewards = pong.play_games(1, 1, game_agent=my_model, save_moves=False, use_turtle=False,
                                                     show_moves=False)
            fitness.append(sum(rewards))
            #print(n, fitness, noise[-1])
        ranked_fittness = centered_ranker(np.array(fitness, dtype=float))
        print(f"Max fitness {max(fitness)} Average fitness {np.average(fitness)}")
        weighted_average_noise = np.dot(ranked_fittness, noise)
        theta += weighted_average_noise * lr
        set_model_weights(my_model,theta)
        my_model.save("model_saved"+str(generation))

    print("awe")

    # train_model_with_feedback_save_n_times("model_saved", n=5)
    # frame_stack, moves, rewards = pong.play_games(21, 21, show_moves=True, save_moves=False, use_turtle=True)
    # play_random=True
    # decision_object=None
    # games_to_play=2000
    # points_per_game=21
    # single_player=True
    # frame_stack, moves, rewards=pong.play_games(games_to_play, points_per_game, show_moves=False)

    # good_moves=rewards[rewards>0.05]
    # bad_moves=rewards[rewards<-0.05]

    # print(good_moves.size)
    # print(bad_moves.size)
    pass


main()

# def keep_learning()
# load_data_make_train_play()
# make_train_and_return_model()
# main()
# frame_stack, moves, rewards=load_saved_data()
# print(len(moves))
# print(frame_stack[0])
# print(frame_stack[100])
# print(frame_stack[200])
# print(frame_stack[300])
# print(frame_stack[400])
# # print(frame_stack[5])
# # print(frame_stack[6])
# # print(frame_stack[7])
# # print(frame_stack[8])
# # print(frame_stack[9])
# # print(frame_stack[10])
# print(moves[0:200])
# # print(len([x for x in rewards if x==1]))
