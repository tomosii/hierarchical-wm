class Periodic:
    def __init__(self, period):
        self.period = period
        self.counter = 0

    def __call__(self, step):
        if step % self.period == 0:
            return True
        return False
