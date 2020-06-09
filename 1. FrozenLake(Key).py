import gym
from gym.envs.registration import register
import _Getch

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
    key = _Getch.inkey()
    if key not in _Getch.arrow_keys.keys():
        print("Gema Aborted!")
        break

    action = _Getch.arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render() #현재 상태

    print("state : {0}, action : {1}, reward : {2}, info : {3}".format(state, action,reward,info))
    if done:
        print("Finished with reward {0}".format(reward))
        break
          
