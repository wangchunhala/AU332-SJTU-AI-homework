import numpy as np
import pandas as pd
from math import log,exp,pow

class Agent:
    ### START CODE HERE ###

        def __init__(self, actions):

            self.actions = actions
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        def choose_action(self, observation,n,re):
            self.check(observation)
            if(n<70):
             e=0.1
            elif(100>n>=70):
             e=1-1/(pow(1.001,n)+2)
            elif(n>=100):
             e=0.99
            elif (n >= 200):
             e = 1
            if np.random.uniform() < e:
                    state_action = self.q_table.loc[observation, :]
                    action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:

                        state_action = self.q_table.loc[observation, :]
                        action = np.random.choice(state_action[state_action >=0].index)


            return action

        def learn(self, s, a, r, s_):
            self.check(s_)
            q_predict = self.q_table.loc[s, a]
            if s_ != 'terminal':
                q_target = r + 0.7 * self.q_table.loc[s_, :].max()
            else:
                q_target = r

            self.q_table.loc[s, a] +=0.1 * (q_target - q_predict)

        def check(self, state):
            if state not in self.q_table.index:
                self.q_table = self.q_table.append(
                    pd.Series(
                        [0] * len(self.actions),
                        index=self.q_table.columns,
                        name=state,
                    )
                )

class EnvModel:
    def __init__(self, actions):
        self.actions = actions
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    def remember(self, s, a, r, s_):
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.actions),
                    index=self.database.columns,
                    name=s,
                ))
        self.database.set_value(s, a, (r, s_))

    def sample(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.loc[s].dropna().index)
        return s, a

    def get(self, s, a):
        r, s_ = self.database.loc[s, a]
        return r, s_