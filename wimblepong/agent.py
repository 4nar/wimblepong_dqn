import time
import numpy as np
import collections
import torch
import cv2



Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._obs_buffer = collections.deque(maxlen=2)

        self._reset()
        self.count_ep = 0

    def preprocess(self, frame):
        if frame.size == 200 * 200 * 3:
            img = np.reshape(frame, [200, 200, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

    def _reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        prep = self.preprocess(obs)
        prep.shape = (prep.shape[-1], prep.shape[0], prep.shape[1])
        prep = np.array(prep).astype(np.float32) / 255.0
        self.buffer = np.zeros((4, 84, 84), dtype=np.float32)
        self._obs_buffer.append(prep)
        self.state = self.buffer
        self.total_reward = 0.0


    def play_step(self, net, epsilon=0.0,  device="cpu"):
        print(epsilon)
        done_reward = None
        if np.random.random() < epsilon:
            print('random')
            action = self.env.action_space.sample()
        else:
            print('not random')
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            print('q_vals_v' , q_vals_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        print(action)
        # action1 = np.random.choice([0, 1, 2])


        def step_(self, action):
            total_reward = 0.0
            done = None

            skip = 4
            for _ in range(skip):
                ob1, rew1, done, info = self.env.step(action)
                self._obs_buffer.append(ob1)
                total_reward += rew1
            max_frame = np.max(np.stack(self._obs_buffer), axis=0)
            prep = self.preprocess(max_frame)
            prep.shape = (prep.shape[-1], prep.shape[0], prep.shape[1])
            prep = np.array(prep).astype(np.float32) / 255.0
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = prep
            return self.buffer, total_reward, done, info

        new_state, reward, is_done, _ = step_(self, action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        self.env.render()
        if is_done:
            self.count_ep += 1
            done_reward = self.total_reward
            self._reset()
            if self.count_ep % 5 == 4:
                self.env.switch_sides()

        return done_reward, is_done