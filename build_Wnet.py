import ncut_loss
import tensorflow as tf


class Wnet(tf.keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        input_shape,

    ):
        super(Wnet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.neighbour_filter = ncut_loss.neighbor_filter(input_shape)

    def compile(
        self,
        optimizer,
        loss_fn_segmentation,
        loss_fn_reconstruction
    ):
        super(Wnet, self).compile()

        self.optimizer = optimizer
        self.loss_fn_segmentation = loss_fn_segmentation
        self.loss_fn_reconstruction = loss_fn_reconstruction #keras.losses.MeanSquaredError()
        
    def call(self, inputs, training=False):
      output = self.decoder(self.encoder(inputs))
      return output

    @tf.function
    def test_step(self,batch_data,reconstruction_loss_weight):
        image= batch_data

        result_encoder = self.encoder(image)
        result_decoder = self.decoder(result_encoder)
        loss_decoder = self.loss_fn_reconstruction(image,result_decoder)
        loss_encoder = self.loss_fn_segmentation(image,result_encoder,self.neighbour_filter)

        return {
            "loss_encoder": loss_encoder,
            "loss_decoder": loss_decoder,
        }

    @tf.function
    def train_step(self, batch_data,reconstruction_loss_weight):
        image = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
          
          result_encoder = self.encoder(image)
          result_decoder = self.decoder(result_encoder)
          loss_decoder = reconstruction_loss_weight*self.loss_fn_reconstruction(image,result_decoder)
          loss_encoder = self.loss_fn_segmentation(image,result_encoder,self.neighbour_filter)
   
          
          
        grads_encoder_1 = tape.gradient(loss_encoder, self.encoder.trainable_variables)
        grads_2 = tape.gradient(loss_decoder, self.encoder.trainable_variables+self.decoder.trainable_variables)
        
        ###grads_decoder = tape.gradient(loss_decoder, self.decoder.trainable_variables)

   
        self.optimizer.apply_gradients(
            zip(grads_encoder_1, self.encoder.trainable_variables)
        )
        self.optimizer.apply_gradients(
            zip(grads_2, self.encoder.trainable_variables+self.decoder.trainable_variables)
        )
        
        ##self.optimizer.apply_gradients(
        ##    zip(grads_decoder, self.decoder.trainable_variables)
        ##)

        return {
            "loss_encoder": loss_encoder,
            "loss_decoder": loss_decoder,
        }
