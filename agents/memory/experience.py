import random
from collections import namedtuple

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward'))


class ExperienceReplay(object):

    def __init__(self, capacity, experience):
        self.capacity = capacity
        self.experience = experience
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, as_tuple=False):
        samples = random.sample(self.memory, batch_size)
        if as_tuple:
            return self.experience(*zip(*samples))
        else:
            return samples

    def __len__(self):
        return len(self.memory)
