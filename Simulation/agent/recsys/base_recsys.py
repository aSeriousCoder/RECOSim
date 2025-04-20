from abc import ABCMeta,abstractmethod  


class BaseRecSys():
    __metaclass__ = ABCMeta

    @abstractmethod
    def recommend_post(self):
        pass

    @abstractmethod
    def recommend_friends(self):
        pass

    @abstractmethod
    def prepare(self):
        pass