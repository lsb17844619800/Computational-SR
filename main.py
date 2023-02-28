import tensorflow as tf 

import os
import gc
import argparse
import run


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # gets rid of avx/fma warning

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Train the model', default=True)
    parser.add_argument('--test', help='Run tests on the model', default=False)
    parser.add_argument('--export', help='Export the model as .pb', default=True)
    parser.add_argument('--fromscratch', help='Load previous model for training',default=False)
    parser.add_argument('--finetune', help='Finetune model on General100 dataset',default=False)
    parser.add_argument('--small', help='Run FSRCNN-small', default=False)
    
    parser.add_argument('--scale', type=int, help='Scaling factor of the model', default=4)
    parser.add_argument('--batch', type=int, help='Batch size of the training', default=10)
    parser.add_argument('--epochs', type=int, help='Number of epochs during training', default=20)
    parser.add_argument('--image', help='Specify test image', default="./images/butterfly.png")
    parser.add_argument('--lr', type=float, help='Learning_rate', default=0.001)
    parser.add_argument('--d', type=int, help='Variable for d', default=56)
    parser.add_argument('--s', type=int, help='Variable for s', default=12)
    parser.add_argument('--m', type=int, help='Variable for m', default=4)
    
    parser.add_argument('--traindir', help='Path to train images',default='traindir/inputs')
    parser.add_argument('--finetunedir', help='Path to finetune images',default='finetunedir')
    parser.add_argument('--validdir', help='Path to validation images',default='validdir/inputs')

    args = parser.parse_args()

    # CHECK
    """
    if args.train == True:
        args.export = False

    if args.export == True:
        args.train = False
    """
    # INIT
    scale = args.scale
    fsrcnn_params = (args.d, args.s, args.m) 
    traindir = args.traindir

    small = args.small

    lr_size = 10
    if(scale == 3):
        lr_size = 7
    elif(scale == 4):
        lr_size = 6
        
    hr_size = lr_size * scale
    
    # FSRCNN-small
    if small:
        fsrcnn_params = (32, 5, 1)

    # Set checkpoint paths for different scales and models
    ckpt_path = ""
    if scale == 2:
        ckpt_path = "./CKPT_dir/x2/"
        if small:
            ckpt_path = "./CKPT_dir/x2_small/"
    elif scale == 3:
        ckpt_path = "./CKPT_dir/x3/"
        if small:
            ckpt_path = "./CKPT_dir/x3_small/"
    elif scale == 4:
        ckpt_path = "./CKPT_dir/x4/"
        if small:
            ckpt_path = "./CKPT_dir/x4_small/"
    else:
        print("Upscale factor scale is not supported. Choose 2, 3 or 4.")
        exit()
    
    # Set gpu 
    config = tf.ConfigProto() #log_device_placement=True
    config.gpu_options.allow_growth = False

    # Create run instance
    run = run.run(config, lr_size, ckpt_path, scale, args.batch, args.epochs, args.lr, args.fromscratch, fsrcnn_params, small, args.validdir)

    if args.train:
        # if finetune, load model and train on general100
        if args.finetune:
            traindir = args.finetunedir

        run.train(traindir)
        gc.collect()

    if args.test:
        run.testFromPb(args.image)
        gc.collect()

    if args.export:
        run.export()
        gc.collect()
    
    print("I ran successfully.")