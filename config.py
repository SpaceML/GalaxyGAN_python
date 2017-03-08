class Config:
    data_path = "/Users/ruru/Desktop/10_jpg"
    model_path = "" #"./datasets/facades/checkpoint/model_100.ckpt"
    output_path = "/Users/ruru/Desktop/10_jpg"

    img_size = 424
    adjust_size = 500
    train_size = 424
    img_channel = 3
    conv_channel_base = 64

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 200
    L1_lambda = 100
    save_per_epoch=1
