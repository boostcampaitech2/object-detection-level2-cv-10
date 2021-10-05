class MetricDict:
    def __init__(self, train_process):
        self.dummy = {x: {} for x in train_process}
        self.iter = 0
        self.__dict__ = self.dummy.copy()

    def average(self):
        pass

    def reset(self):
        self.iter = 0
        self.__dict__ = self.dummy.copy()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def copy(self):
        return self.__dict__.copy()

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return iter(self.__dict__)

class GetConfig:
    def __init__(self, config) -> None:
        for key in config:
            var = config[key]
            if type(var) == str:
                exec(f"self.{key} = '{var}'")
            else:
                exec(f"self.{key} = {var}")
