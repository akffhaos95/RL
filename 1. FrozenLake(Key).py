import gym
from gym.envs.registration import register
import msvcrt #Windows 키 입력

class _Getch:
    def __call__(self):
        return msvcrt.getch()
inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    b'w' : UP,
    b'a' : LEFT,
    b's' : DOWN,
    b'd' : RIGHT
}

register(
    id = 'FrozenLake-v3',
    entry_point= "gym.envs.toy_text:FrozenLakeEnv",
    kwargs={
        'map_name' : '4x4',
        'is_slippery':False
        }
)

env = gym.make('FrozenLake-v3')
env.render()

while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print("Gema Aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render() #현재 상태

    print("state : {0}, action : {1}, reward : {2}, info : {3}".format(state, action,reward,info))
    if done:
        print("Finished with reward {0}".format(reward))
        break
          
