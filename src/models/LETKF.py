from config import config
from data.dataset import RankineData

class LETKF:
    def __init__(self):
        self.name = 'LETKF'
        self.emsemble_size = config.emsemble_size
        self.data = RankineData(config.number_of_prior)

    def make_localization_matrix(sigma):
        pass

    def preparation_for_analysis(self):
        pass

    def analysis(self):
        pass

    def run(self):
        pass

    