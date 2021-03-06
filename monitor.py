import neptune.new as neptune

class NeptuneLog:
    def __init__(self, prefix='default'):
        self.prefix = prefix
        self.logger = neptune.init(project='viettoan.bk1997/GameDetector',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2OGIwOGM2Yi01NTM1LTRkYjUtODYxYy00ZmMyN2UzMWQ5N2QifQ==') # your credentials

    def log_metrics(self, metrics):
        for metric_name, metric_val in metrics:
            metric_name = '{}/{}'.format(self.prefix, metric_name)
            self.logger[metric_name].log(metric_val)
            
    def log_scalars(self, metrics):
        for metric_name, metric_val in metrics:
            metric_name = '{}/{}'.format(self.prefix, metric_name)
            self.logger[metric_name] = metric_val