

class Patient:

    def __init__(self,path):
        self.path = path
        self.images_path = f'{self.path}/images/'
        self.seg_path    = f'{self.path}/seg/'


    def read_images(self):
        pass

    def read_segmentations(self):
        pass