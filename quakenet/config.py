class Config(object):
  def __init__(self):
    self.learning_rate = 0.001
    self.batch_size = 128
    self.win_size = 1001
    self.n_traces = 3
    self.display_step = 50
    self.n_threads = 2
    self.n_epochs = None
    self.regularization = 1e-3
    self.n_clusters = None

    # Number of epochs, None is infinite
    n_epochs = None
