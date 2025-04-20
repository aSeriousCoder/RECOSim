import pickle


class MetricsService():
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self._data = {}

    def save(self):
        self._logger.debug('Saving Metrics ...')
        saving_dir = self._config.simulation_result_dir
        with open(f'{saving_dir}/metrics.pkl', 'wb') as file:
            pickle.dump(self._data, file)

    def load(self):
        self._logger.debug('Loading Metrics ...')
        saving_dir = self._config.simulation_result_dir
        with open(f'{saving_dir}/metrics.pkl', 'rb') as file:
            self._data = pickle.load(file)

    def append(self, key, value):
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(value)

