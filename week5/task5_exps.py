import os
import random

def mkdirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

mkdirs('task5_results/')

batch_sizes  = [16, 32, 64, 128]
epochs = [10, 20, 50, 70]
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'SGDNesterov']
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
momentums = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
activations = ['relu', 'tanh', 'sigmoid']
flips = [True, False]
zooms = [0, 0.1, 0.2]
width_ranges = [0, 0.1, 0.2, 0.3]
height_ranges =  [0, 0.1, 0.2, 0.3]
rotations = [0, 5, 10, 20, 30]
dropouts = [0, 0.25, 0.5, 0.75]
topologies = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
weight_decays = [0.0, 0.00001, 0.0001, 0.001]
batch_norms = [True, False]
lr_decays = [0.0, 0.0001, 0.0001, 0.001]

run = 2

i = 0
while True:
    batch_size = random.choice(batch_sizes)
    n_epochs = random.choice(epochs)
    optimizer = random.choice(optimizers)
    lr = random.choice(learning_rates)
    momentum = random.choice(momentums)
    activation = random.choice(activations)
    flip_horizontal = random.choice(flips)
    zoom = random.choice(zooms)
    w_shift = random.choice(width_ranges)
    h_shift = random.choice(height_ranges)
    rotation = random.choice(rotations)
    dropout = random.choice(dropouts)
    topology = random.choice(topologies)
    batch_norm = random.choice(batch_norms)
    wd = random.choice(weight_decays)
    lr_decay = random.choice(lr_decays)

    # build call
    call = "python w5_task5.py"
    call += " -test "+topology
    call += " -train_folder train_400"
    call += " -batch_size "+str(batch_size)
    call += " -n_epochs "+str(n_epochs)
    call += " -opt "+optimizer
    call += " -lr "+str(lr)
    call += " -momentum "+str(momentum)
    call += " -act "+activation
    if flip_horizontal:
        call += " --h_flip"
    call += " -zoom "+str(zoom)
    call += " -w_shift "+str(w_shift)
    call += " -h_shift "+str(h_shift)
    call += " -rotation "+str(rotation)
    call += " -drop "+str(dropout)
    if batch_norm:
        call += " --batch_norm"
    call += " -wd "+str(wd)
    call += " -lr_decay "+str(lr_decay)

    call += " > task5_results/run"+str(run)+"_"+str(i)+".txt 2>&1"

    os.system(call)
    print(call)

    i += 1