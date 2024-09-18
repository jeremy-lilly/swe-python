import time

class Timer:
    _start = 0.0
    _end = 0.0
    _time = 0.0

    def start(self):
        self._start = time.time()

    def stop(self):
        self._end = time.time()

    def reset(self):
        self._time = 0

    def get_time(self):
        self._time += self._end - self._start
        return self._time

    def get_time_reset(self):
        _temp_time = self.get_time()
        self.reset()
        return _temp_time