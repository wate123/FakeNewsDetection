
class FeatureGenerator(object):

    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def process_and_save(self, data, header):
        '''
            input:
                data: dict
            generate features and save them into a json
        '''
        pass

    def read(self, header):
        '''
            read the feature matrix from a json
        '''
        pass