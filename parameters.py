import torch

class Parameters:
    def __init__(self):
        self.model='pixelcnnpp'
        self.n_channels=128             #'Number of channels for gated residual convolutional layers.')
        self.n_res_layers=1             #'Number of Gated Residual Blocks.')
        self.norm_layer=True            #'Add a normalization layer in every Gated Residual Blocks.')
        self.device = torch.device('cuda')
        self.drop_rate=.5
        self.n_logistic_mix=10          #'Number of of mixture components for logistics output.')
        #self.kernel_size=5             #'Kernel size for the gated residual convolutional blocks.')
        #self.n_out_conv_channels=128   # 1024 #'Number of channels for outer 1x1 convolutional layers.')

        # action
        self.train=True            #'Train model.')
        self.evaluate=True         #'Evaluate model.')
        self.generate=True         #'Generate samples from a model.')
        self.output_dir=r''        #'Path to model to restore.')
        self.restore_file=r'C:\Users\W.Rogers\PixelCNN\vi\results\pixelcnnpp\2020-10-25_00-00-00\checkpoint_1856.pt'
        #self.restore_file=False
        self.restore_opt=False
        self.seed=0                  #'Random seed to use.')
        #self.cuda='0,1,2,3,4,5,6,7' #'Which cuda device to use.')
        #self.cuda='1,2,3,4,5,6,7'    #'Which cuda device to use.')
        self.cuda='0,1,2'           #'Which cuda device to use.')

        # data params
        self.dataset='dataset'
        self.n_cond_classes=None      #'Number of classes for class conditional model.')
        self.n_bits=8                 #'Number of bits of input data.')
        self.image_dims=(1,3,64,64) #'Dimensions of the input data.')
        self.crop = True
        #self.output_dims=(3,64,64) #'Dimensions of the output data')
        self.scale_input=2
        self.shift_input=-1

        # optimizer
        self.optimizer='adam'
        self.lr=.000008           #'Learning rate.')
        #self.lr=0.001
        self.decay=.95          # Beta 1 for Adam or momentum for SGD

        # training paramS
        self.batch_size=18      #'Training batch size.')
        self.n_epochs=4096        #'Number of epochs to train.')
        self.step=0             #'Current step of training (number of minibatches processed).')
        self.start_epoch=0      #'Starting epoch (for logging; to be overwritten when restoring file.')
            # by epoch
        self.save_interval=256   #'How often to save samples.')
        self.eval_interval=4     #'How often to evaluate
        self.gen_interval=9999    # How often to generate and save samples.')
             # by step
        #self.log_interval=8*14     #'How often to show statistics')
        self.log_interval=11*4
        self.train_gen=True    # Display the training generation (stops training in terminal)

        # Scheduler
        self.scheduler = None #'exponential'
        self.polyak=0.9995      #'Polyak decay for parameter exponential moving average on RMSProp
        #self.lr_decay=0.999995  #'Learning rate decay, applied every step of the optimization.')
        self.lr_decay =0.99995

        # cosine annealing
        self.eta_min = .0009

        # generation param
        self.n_samples=1        #'Number of samples to generate.')
        self.ymin=2

        # initializers
        initializers = [None,
                        torch.nn.init.uniform_,         #1
                        torch.nn.init.normal_,          #2
                        torch.nn.init.constant_,        #3
                        torch.nn.init.ones_,            #4
                        torch.nn.init.zeros_,           #5
                        torch.nn.init.dirac_,           #6
                        torch.nn.init.xavier_uniform_,  #7
                        torch.nn.init.xavier_normal_,   #8
                        torch.nn.init.kaiming_uniform_, #9
                        torch.nn.init.kaiming_normal_,  #10
                        torch.nn.init.orthogonal_]      #11

        self.init=initializers[1]








