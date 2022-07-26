from numpy import ndarray
import numpy as np


class ImageTransformPipeline:
    """
    Pipeline to transform image data before use in models
    
    Pipeline:
           
        1. Log-transform the data: y_ijk -> log(1 + y_ijk)
        2. Subtract mean across each channel
        3. Normalise standard deviation in each channel
       
    """
    
    
    def __init__(self, raw_data: ndarray):
        
        self.raw_data = raw_data.astype(float)
        self.channel_mean = self.raw_data.mean((0, 1))
        
    def transform(self):
        
        data = np.log(1 + self.raw_data)
        self.channel_mean = data.mean((0, 1))[None, None, :]
        data -= self.channel_mean
        self.channel_std = data.std((0, 1))[None, None, :]
        data /= self.channel_std
            
        return data
    
    def reverse_transform(self, new_data: ndarray):
        
        data = new_data.copy()
        data *= self.channel_std
        data += self.channel_mean
        
        data = np.clip((np.exp(data) - 1), 0, 255).astype('uint8')
                
        return data