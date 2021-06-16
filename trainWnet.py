import sys
import yaml
from getData import get_generator
from build_Unet import build_Unet
from build_Wnet import Wnet
from ncut_loss import compute_soft_ncuts
import numpy as np
import tensorflow as tf
from trainer import training 
import os

def main(arg):

    EXP_FOLDER= arg[0]

    print(arg[0])

    with open(EXP_FOLDER+"/custom.yaml", 'r') as stream:
        custom_data = yaml.safe_load(stream)

    INPUT_DIM = int(custom_data['INPUT']['DIM'])
    TRAIN_DATASET = custom_data["TRAIN_DATASET"]
    TEST_DATASET = custom_data["TEST_DATASET"]

    SUBCATS = custom_data["SUBCATS"]
    K = int(custom_data["MODEL"]["K"])
    STAGES = (np.arange(1,int(custom_data["MODEL"]["STAGES"])+1))
    FILTERS = int(custom_data["MODEL"]["FILTERS"])
    MAX_ITER = int(custom_data["SOLVER"]["MAX_ITER"])
    IMS_PER_BATCH = int(custom_data["SOLVER"]["IMS_PER_BATCH"])
    BASE_LR = float(custom_data["SOLVER"]["BASE_LR"])
    STEPS = [int(custom_data["SOLVER"]["STEPS"][0]),int(custom_data["SOLVER"]["STEPS"][1])]
    
    CHECKPOINT_PERIOD = int(custom_data["CHECKPOINT_PERIOD"])
    IMG_PERIOD =int( custom_data["IMG_PERIOD"])
    TEST_PERIOD = int(custom_data["TEST_PERIOD"])

    RECONSTRUCTION_LOSS_WEIGHT = int(custom_data["RECONSTRUCTION_LOSS_WEIGHT"])

    generator_train = get_generator(TRAIN_DATASET,INPUT_DIM,IMS_PER_BATCH,SUBCATS)
    generator_test = get_generator(TEST_DATASET,INPUT_DIM,IMS_PER_BATCH,SUBCATS)
    

    encoder = build_Unet(K=K,stages = STAGES,filters = FILTERS,type='encoder',input_size=INPUT_DIM)
    decoder = build_Unet(K=K,stages = STAGES,filters = FILTERS,type='decoder',input_size=INPUT_DIM)
    wn = Wnet(encoder,decoder,(INPUT_DIM,INPUT_DIM))

    ## Load weights
    start_iter=0
    ckpts=[]

    if('checkpoint' in os.listdir(EXP_FOLDER)):
        with open(EXP_FOLDER+"/checkpoint", 'r') as stream:
            try:
                ckpts = yaml.safe_load(stream)
                last_ckpt = ckpts["model_checkpoint_path"]
                wn.load_weights(EXP_FOLDER+'/'+last_ckpt)
                start_iter = int(last_ckpt.replace('ckpt',''))

            except yaml.YAMLError as exc:
                print(exc)
                  
    print('Starting from iter : ', start_iter)

    

    # Compile the model
    wn.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=BASE_LR),
        loss_fn_segmentation = compute_soft_ncuts,
        loss_fn_reconstruction = tf.keras.losses.MeanSquaredError()
    )

    training(model=wn,train_dataset=generator_train,test_dataset=generator_test,max_iter=MAX_ITER,start_iter=start_iter,base_lr=BASE_LR,ckpt_freq=CHECKPOINT_PERIOD,img_freq=IMG_PERIOD,dir_path=EXP_FOLDER,solver_steps=STEPS,test_freq=TEST_PERIOD,reconstruction_loss_weight=RECONSTRUCTION_LOSS_WEIGHT)
   
  

if __name__ == "__main__":
    
    main(sys.argv[1:])


