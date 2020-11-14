
#using turtle for 100 games takes 37.4 seconds (14.1% hit rate using random moves)
#using my own turtle class for 100 games takes 2.4 seconds (we use this for training and testing) (14.2% hit rate using random moves)
import random
import time
import numpy as np
import pong
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense



def generate_random_moves_and_save():
    frame_stack, moves, rewards=pong.play_games(10000, 21,single_player=True ,show_moves=False,game_agent=None, save_moves=True, use_turtle=False)
    print(len(frame_stack))
generate_random_moves_and_save()
def make_model_play(model_name, games,show_moves,save_moves=False):
    model=load_model(model_name)
    frames, moves, rewards = pong.play_games(games, 10, show_moves=show_moves, save_moves=save_moves,game_agent=model)
    return frames, moves, rewards
#make_model_play("model_saved", 10, False, False)

def train_model_with_feedback_save_n_times(model, n=5):
    model_name=model
    model=load_model(model)
    for x in range(n):
        frames, moves,rewards=make_model_play(model_name, 1000,show_moves=False)
        print("misses {} hits{}".format(sum([x for x in rewards if x==-1]),sum([x for x in rewards if x==1])))
        trained_model=train_model(model,frames,moves,rewards,20,200)
        model.save("model_".format((n), ))
        model_name="model_"+str(n)
        print("SAVED MODEL" + str(n))
def load_saved_data():
    with open("frame_stack.npy","rb") as f:
        frame_stack=np.load(f)
    with open("moves.npy","rb") as f:
        moves=np.load(f)
    with open("rewards.npy","rb") as f:
        rewards=np.load(f)
    return frame_stack, moves, rewards
def make_a_model(dim_inputs=9, layer_1_nodes=300, layer_2_nodes=50, output_nodes=1):
    model=Sequential()
    model.add(Dense(layer_1_nodes, input_dim=dim_inputs, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_data_make_train_play():
    frame_stack, moves, rewards=load_saved_data()
    model=make_a_model(9,300,25, 1)
    rewards=pong.discount_rewards(rewards, 0.95)
    trained_model=train_model(model, frame_stack,moves,rewards,5,100)
    pong.play_games(10, 21,single_player=True ,show_moves=True,game_agent=trained_model, save_moves=False)

def train_model(model, frame_stack, moves, rewards, epoch_count, batch_size_specified):
    training_set_moves=[]
    training_set_frames=[]
    #get rid of moves and frames that won't benefit us
    data=[]
    for x in range(len(rewards)):
        if(rewards[x]>0.5):
            #data.append([frame_stack[x]]+moves[x])
            training_set_frames.append(frame_stack[x])
            training_set_moves.append(moves[x])
    training_set_frames=np.array(training_set_frames,dtype=float)
    training_set_moves=np.array(training_set_moves, dtype=float)
    training_set_moves=training_set_moves.reshape(training_set_moves.shape[0],1)

    print("Frames dimensions {}".format((len(training_set_frames),training_set_frames[0].shape)))
    #print(training_set_frames[1])
    #print(training_set_frames[2])
    #print(training_set_frames[3])
    #print(training_set_frames[4])
    print("Moves dimensions {}".format(training_set_moves.shape))
    #print(training_set_moves)
    print("Training set has {} items".format(len(training_set_moves)))
    #good_moves=rewards[rewards>0.05]
    #bad_moves=rewards[rewards<-0.05]
    print("model made")
    model.fit(training_set_frames, training_set_moves, epochs=epoch_count, batch_size=batch_size_specified)
    return model
def make_train_and_return_model():
    frame_stack, moves, rewards = pong.play_games(5000,21, show_moves=False, save_moves=True)
    model=make_a_model(9, 50, 10, 1)
    model=train_model(model, frame_stack, moves, rewards,10, 200 )
    model.save("model_saved")
    #print(frame_stack[1].reshape(1,9))
    #print(model.predict(frame_stack[1].reshape(1,9)))
    pong.play_games(10,20,game_agent=model)
#make_train_and_return_model()
def load_model_train_on_play_data_return_results(model_name,games_to_play=100):
    model=load_model("model_saved")
    frames, moves, rewards=load_saved_data()
    model=train_model(model,frames,moves,rewards,10,3)
    frames, moves, rewards = pong.play_games(10, 10, show_moves=False, save_moves=False,
                                             game_agent=model)
    return frames, moves, rewards
def main():
    #train_model_with_feedback_save_n_times("model_saved", n=5)

    # play_random=True
    # decision_object=None
    # games_to_play=2000
    # points_per_game=21
    # single_player=True
    # frame_stack, moves, rewards=pong.play_games(games_to_play, points_per_game, show_moves=False)

    #good_moves=rewards[rewards>0.05]
    #bad_moves=rewards[rewards<-0.05]

    #print(good_moves.size)
    #print(bad_moves.size)
    pass

main()


# def keep_learning()
#load_data_make_train_play()
#make_train_and_return_model()
#main()
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