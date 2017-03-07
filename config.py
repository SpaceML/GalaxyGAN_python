class Config:
    #used for training	
    data_path = "../data"
    save_path = "../model"

    #if you are not going to train from the very beginning, change this path to the existing model path
    model_path = ""#"../model/model.ckpt"

    start_epoch = 0
    output_path = "../result"

    #used GPU
    use_gpu = 1

    #changed to FITs, mainly refer to the size
    img_size = 424
    train_size = 424
    img_channel = 3
    conv_channel_base = 64

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 100
    L1_lambda = 100
    sum_lambda = 0####
    save_per_epoch=10
