class Config:
    data_path = "./datasets/facades"
    model_path = ""#"./datasets/facades/checkpoint/model_100.ckpt"
    output_path = "./results"

    img_size = 256
    adjust_size = 286
    train_size = 256
    img_channel = 3
    conv_channel_base = 64

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 200
    L1_lambda = 100
    save_per_epoch=5
