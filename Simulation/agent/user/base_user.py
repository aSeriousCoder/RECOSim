from abc import ABCMeta,abstractmethod  


class BaseUser():
    __metaclass__ = ABCMeta

    @abstractmethod
    def update_ugt_state(self):
        pass

    @abstractmethod
    def browse(self, news_recommendation):
        pass

    @abstractmethod
    def follow(self, friends_recommendation, user_browse_actions):
        pass
