#64 hits, 210 misses

# pong game taken from https://www.codegrepper.com/code-examples/python/create+pong+game+with+python
# and adapted to a 1 player game and flipped sideways.
import turtle
import random
import time
import numpy as np

class my_turtle():
    def __init__(self, dx,dy,x, y ):
        self.dx=dx
        self.dy=dy
        self.x=x
        self.y=y

    def xcor(self):
        return self.x
    def ycor(self):
        return self.y
    def setx(self,x):
        self.x=x
    def sety(self,y):
        self.y=y
    def goto(self,x,y):
        self.x=x
        self.y=y


def get_turtle():
    paddle_a = turtle.Turtle()
    paddle_a.speed(0)
    paddle_a.shape("square")
    paddle_a.color("black")
    paddle_a.shapesize(stretch_wid=1, stretch_len=5)
    paddle_a.penup()
    paddle_a.goto(0, -250)

    paddle_b = turtle.Turtle()
    paddle_b.speed(0)
    paddle_b.shape("square")
    paddle_b.color("black")
    paddle_b.shapesize(stretch_wid=1, stretch_len=5)
    paddle_b.penup()
    paddle_b.goto(250, 250)
    wn = turtle.Screen()
    # wn.title("Ping Pong !!!")
    wn.bgcolor("white")
    wn.setup(width=800, height=600)
    wn.tracer(0)
    ball = turtle.Turtle()
    ball.speed(0)
    ball.shape("circle")
    ball.color("black")
    ball.penup()
    ball.goto(0, 0)
    ball.dx = 0.6  # Can change the speed according to the speed of your system.
    ball.dy = 0.6  # Can change the speed according to the speed of your system.
    # Pen
    pen = turtle.Turtle()
    pen.speed(0)
    pen.shape("square")
    pen.color("black")
    pen.penup()
    pen.hideturtle()
    pen.goto(0, 260)
    pen.write("Player A: 0  Player B: 0", align="center", font=("Courier", 24, "normal"))
    return paddle_a,paddle_b, ball, wn



def discount_rewards(r, gamma=0.99):
    "”” take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # if the game ended (in Pong), reset
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)  # normalizing the result
    discounted_r /= np.std(discounted_r)  # idem using standar deviation
    return discounted_r




def play_games(games_to_play, points_per_game,single_player=True ,show_moves=True,game_agent=None, save_moves=False, use_turtle=True):
    hit_reward = 1
    if(use_turtle==True):
        paddle_a, paddle_b, ball, wn=get_turtle()
    else:
        paddle_a=my_turtle(0,0,0,0)
        paddle_b=my_turtle(0,0,0,0)
        ball = my_turtle(0, 0, 0, 0)
        paddle_a.goto(0, -250)
        paddle_b.goto(250, 250)
        ball.goto(0, 0)
        ball.dx = 0.6
        ball.dy = 0.6
    time_score = 0
    # Score
    score_a = 0
    score_b = 0
    last_move=0
    # Paddle A

    """
    # Paddle B
   
    """
    # Ball

    def paddle_a_right():
        if (paddle_a.xcor() >= 350):
            return
        x = paddle_a.xcor()
        x += 15
        paddle_a.setx(x)
        #print("Paddle position {0} Ball x coord: {1} ball y coord: {2}".format((paddle_a.xcor(), paddle_a.ycor()),ball.xcor(), ball.ycor()))
    def paddle_a_left():
        if (paddle_a.xcor() <= -350):
            return
        x = paddle_a.xcor()
        x -= 15
        paddle_a.setx(x)
        #print("Paddle position {0} Ball x coord: {1} ball y coord: {2}".format((paddle_a.xcor(), paddle_a.ycor()), ball.xcor(), ball.ycor()))
    # Keyboard bindings
    if(single_player==False):
        wn.listen()
        wn.onkeypress(paddle_a_left, "a")
        wn.onkeypress(paddle_a_right, "d")
    # a=1
    # frame stack will keep the 20 previous time-steps
    #frame_stack=[ ...., [ball.x, ball.y, paddle.x, paddle.y, ball.x-paddle.x,ball.y-paddle.y, ball.x-paddle.x + ball.y-paddle.y, left(0)_or_right(1),
    # game_in_progress=1 else 0, previous configuration of all the previous data, ]] all noramalised to [-1,1]
    frame_stack=[]
    #moves=[0-1, 0-1,...] the moves made during the above frames
    moves=[]
    #rewards=[] the rewards based off the above frames
    rewards=[]
    ball_hits=0
    ball_misses=0
    score_a = 0
    score_b = 0
    bounce_counter=0
    previous_frame=np.array(np.zeros(9))
    st=time.time()
    game_in_progress=False
    last_move=0
    frame_counter=0
    # Main game loop
    if(game_agent==None):
        updates_per_frame=20
    else:
        updates_per_frame=10
    #print("Playing episode {} of {}".format(episode_count+1, episodes_to_play))
    for game_counter in range(games_to_play):
        print("Playing Game {} of {} ".format(game_counter+1,games_to_play ))
        score_a = 0
        score_b = 0
        bounce_counter=0
        game_in_progress=True
        time_score=0
        while game_in_progress==True:
            frame_counter+=1
            # a*=-1
            # if(a==1):
            ball.setx(ball.xcor() + ball.dx)
            ball.sety(ball.ycor() + ball.dy)
            if(show_moves==True and use_turtle==True):
                wn.update()
            #time_score+=0.1
            if(frame_counter%updates_per_frame==0): #only move and record the state every 10 frames
                if(game_agent==None):
                    last_move=float(random.randint(0,1))
                else:
                    last_move=game_agent.predict(previous_frame.reshape(1,9))
                    if(last_move>0.5):
                        last_move=1
                    else:
                        last_move=0
                if (last_move == 0):
                    paddle_a_left()
                else:
                    paddle_a_right()
                moves.append(last_move)
                # print()
                # Move the ball

                # Border checking
                # Top and bottom
                if ball.ycor() > 290:
                    ball.sety(290)
                    ball.dy *= -1
                # Left and right
                if ball.xcor() > 350:
                    ball.dx *= -1
                elif ball.xcor() < -350:
                    ball.dx *= -1

                # Paddle and ball collisions
                if ball.ycor() <= -250 and ball.xcor() < paddle_a.xcor() + 50 and ball.xcor() > paddle_a.xcor() - 50:
                    ##print("We hit the ball! {0} Bounces".format(bounce_counter))             ###!!!
                    ball_hits+=1
                    ball.dy *= -1
                    bounce_counter+=1
                    ball.sety(ball.ycor()+12)
                    rewards.append(1.0)
                elif ball.ycor() < -255:
                    score_b += 1
                    # pen.clear()
                    # pen.write("Player A: {}  Time score: {}".format(score_a, time_score), align="center", font=("Courier", 24, "normal"))
                    # ball must start in a random position, or the network will just learn a pattern
                    ball.goto((random.random() * 350) - 175, 250)
                    ball.dx *= random.choice([1, -1])
                    rewards.append(-1.0)
                    ball_misses+=1
                    # print("We missed the ball. Time score: {0}  Bounces: {1}".format(time_score, bounce_counter))    ###!!!
                    bounce_counter = 0
                    time_score = 0
                    # print("Paddle position {0} Ball x coord: {1} ball y coord: {2}".format((paddle_a.xcor(), paddle_a.ycor()),
                    #                                                                      ball.xcor(), ball.ycor()))
                else:
                    rewards.append(0.0)


                ball_x=ball.xcor()
                ball_y=ball.ycor()
                paddle_x=paddle_a.xcor()
                paddle_y=paddle_a.ycor()
                current_frame=np.array([ball_x, ball_y, paddle_x, paddle_y, ball_x-paddle_x, ball_y-paddle_y, ball_x-paddle_x + ball_y-paddle_y, last_move,1 if game_in_progress==True else 0])
                frame_stack.append(current_frame)
                #print(current_frame)
                #print(last_move)
                #print(rewards[-1])
                previous_frame=current_frame
                if (score_a >= points_per_game or score_b >= points_per_game):
                    game_in_progress = False
                    ### print("Game over ")               ###!!!
                    bounce_counter = 0
    #rewards=discount_rewards(rewards, 0.95)
    print(len(rewards))
    print(len(frame_stack))
    print(len(moves))
    print("Ball Misses {}".format(ball_misses))
    print("Ball Hits {}".format(ball_hits))
    if(save_moves==True):
        with open("move_array.npy", "wb") as f:
            np.save(f,np.array(moves, dtype=np.float64))
        with open("frame_array.npy", "wb") as f:
            np.save(f,np.array(frame_stack, dtype=np.float64))
        with open("reward_array.npy", "wb") as f:
            np.save(f,np.array(rewards, dtype=np.float64))
    print(time.time()-st)
    return frame_stack, moves, rewards
#a,b,c=play_games(100, 21, show_moves=False,use_turtle=False)
# print(len(a[0]))
# print(len(a[1]))
# print(len(a[2]))
# print(list(a[2]))
