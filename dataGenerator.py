from monk import Dataset, BBox
import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,dataset, batch_size=32, dim=(128,128), n_channels=3,shuffle=True):
          
        self.dataset=dataset
        self.dim = dim ###
        self.batch_size = batch_size  ##
        self.list_IDs = np.arange(len(dataset)) ###
        self.n_channels = n_channels ##
        self.shuffle = shuffle ##
        self.labels_to_keep= labels_to_keep
        
        self.get_map_id()
        self.on_epoch_end()
        
        #self.labels = labels
        #self.n_classes = n_classes
        
    def get_map_id(self):
        i=0
        map_id ={}
        for _,imds in enumerate(self.dataset):
            for ann_id,ann in enumerate(imds.anns["polygons"]):
                map_id[i]=[imds.id,ann_id]
                i+=1
        self.map_id = map_id
        
    def load_image(self,ids):
        imds = self.dataset[ids[0]]
        ann = imds.anns["polygons"][ids[1]]
        att =ann.attributes
        img_crop = imds.image.crop(ann.to_bbox()).resize(self.dim)
       
        return(img_crop.rgb)
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=int)
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            print(ID)
            print(self.map_id[ID])
            X[i,] = self.load_image(self.map_id[ID])

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X