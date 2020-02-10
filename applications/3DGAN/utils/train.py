import os

import matplotlib as mpl
mpl.use('Agg')

from utils.ObjectData import ObjectDataset
from net.Generator import Generator 
from net.Discriminator import Discriminator
from utils.VisdomLinePlotter import VisdomLinePlotter
from utils.utils import save_plot_voxels

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage



batch_size = 20
n_workers = 0
learning_rate_G = 0.002
learning_rate_D = 0.0002
beta_1 = 0.5
device = "cuda"
output_dir = "checkpoint_toilet_v2/"
alpha = 0.5
CKPT_PREFIX = "tlet"
SAVE_INTERVAL = 10
EPOCHS = 1000
PRINT_INTERVAL = 10

FAKE_IMG_FNAME = output_dir+'fake_sample_epoch_{:04d}'
REAL_IMG_FNAME = output_dir+'real_sample_epoch_{:04d}'
LOGS_FNAME = 'logs.tsv'

side_len = 32
z_dim = 64

# dataset
object_data = ObjectDataset("data/train_toilet.csv", side_len=side_len)
object_data_loader = DataLoader(object_data, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)

# netowrks
netG = Generator(side_len, z_dim).to(device)
netD = Discriminator(side_len).to(device)


#netG.load_state_dict(torch.load("checkpoint_toilet/tlet_checkpoint_95.pth")["netG"])
#netD.load_state_dict(torch.load("checkpoint_toilet/tlet_checkpoint_95.pth")["netD"])

# criterion
bce = nn.BCELoss()

# optimizers
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate_G, betas=(beta_1, 0.5))
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate_D, betas=(beta_1, 0.5))

# misc
real_labels = torch.ones(batch_size, device=device)
fake_labels = torch.zeros(batch_size, device=device)
fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)

# plotter
plotter = VisdomLinePlotter(env_name="3dgan_train")

def get_noise():
    return torch.randn(batch_size, z_dim, 1, 1, device=device)

def step(engine, batch):

        real = batch
        real = real.to(device)

        # -----------------------------------------------------------
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()

        # train with real
        output = netD(real)
        errD_real = bce(output, real_labels)
        D_x = output.mean().item()
        errD_real.backward()

        # get fake image from generator
        noise = get_noise()
        fake = netG(noise)

        # train with fake
        output = netD(fake.detach())
        errD_fake = bce(output, fake_labels)
        D_G_z1 = output.mean().item()

        errD_fake.backward()

        # gradient update
        errD = errD_real + errD_fake
        optimizerD.step()

        # -----------------------------------------------------------
        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()

        # Update generator. We want to make a step that will make it more likely that discriminator outputs "real"
        output = netD(fake)
        errG = bce(output, real_labels)
        D_G_z2 = output.mean().item()

        errG.backward()

        # gradient update
        optimizerG.step()

        return {
            'errD': errD.item(),
            'errG': errG.item(),
            'D_x': D_x,
            'D_G_z1': D_G_z1,
            'D_G_z2': D_G_z2
        }



trainer = Engine(step)
checkpoint_handler = ModelCheckpoint(output_dir, CKPT_PREFIX, n_saved=10, require_empty=False)
timer = Timer(average=True)

# attach running average metrics
monitoring_metrics = ['errD', 'errG', 'D_x', 'D_G_z1', 'D_G_z2']
RunningAverage(alpha=alpha, output_transform=lambda x: x['errD']).attach(trainer, 'errD')
RunningAverage(alpha=alpha, output_transform=lambda x: x['errG']).attach(trainer, 'errG')
RunningAverage(alpha=alpha, output_transform=lambda x: x['D_x']).attach(trainer, 'D_x')
RunningAverage(alpha=alpha, output_transform=lambda x: x['D_G_z1']).attach(trainer, 'D_G_z1')
RunningAverage(alpha=alpha, output_transform=lambda x: x['D_G_z2']).attach(trainer, 'D_G_z2')

# attach progress bar
pbar = ProgressBar()
pbar.attach(trainer, metric_names=monitoring_metrics)



@trainer.on(Events.ITERATION_COMPLETED)
def print_logs(engine):
    fname = os.path.join(output_dir, LOGS_FNAME)
    columns = ["iteration", ] + list(engine.state.metrics.keys())
    values = [str(engine.state.iteration), ] + \
             [str(round(value, 5)) for value in engine.state.metrics.values()]

    with open(fname, 'a') as f:
        if f.tell() == 0:
            print('\t'.join(columns), file=f)
        print('\t'.join(values), file=f)

    message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                          max_epoch=EPOCHS,
                                                          i=(engine.state.iteration % len(object_data_loader)),
                                                          max_i=len(object_data_loader))
    for name, value in zip(columns, values):
        message += ' | {name}: {value}'.format(name=name, value=value)

    pbar.log_message(message)


# adding handlers using `trainer.on` decorator API
@trainer.on(Events.EXCEPTION_RAISED)
def handle_exception(engine, e):
    if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
        engine.terminate()
        warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

        create_plots(engine)
        checkpoint_handler(engine, {
            'netG_exception': netG,
            'netD_exception': netD
        })

    else:
        raise e

# adding handlers using `trainer.on` decorator API
@trainer.on(Events.EPOCH_COMPLETED(every=PRINT_INTERVAL))
def save_fake_example(engine):
    fake = netG(fixed_noise).reshape(-1, side_len, side_len, side_len)
    plotter.plot_voxels(FAKE_IMG_FNAME.format(engine.state.epoch), fake[0].detach().cpu().numpy(), FAKE_IMG_FNAME.format(engine.state.epoch), savePLY=True)

# adding handlers using `trainer.on` decorator API
@trainer.on(Events.EPOCH_COMPLETED(every=PRINT_INTERVAL))
def save_real_example(engine):
    img = engine.state.batch.reshape(-1, side_len, side_len, side_len)
    plotter.plot_voxels(REAL_IMG_FNAME.format(engine.state.epoch), img[0].detach().cpu().numpy(), REAL_IMG_FNAME.format(engine.state.epoch))


# adding handlers using `trainer.on` decorator API
@trainer.on(Events.EPOCH_COMPLETED)
def print_times(engine):
    pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))
    timer.reset()

# adding handlers using `trainer.on` decorator API
"""
@trainer.on(Events.EPOCH_COMPLETED(every=PRINT_INTERVAL))
def create_plots(engine):
    fake = netG(fixed_noise).reshape(-1, side_len, side_len, side_len)
    save_plot_voxels(fake[0:10], FAKE_IMG_FNAME.format(engine.state.epoch), engine.state.epoch)
"""
# adding handlers using `trainer.add_event_handler` method API
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                          to_save={
                              'netG': netG,
                              'netD': netD
                          })

# automatically adding handlers via a special `attach` method of `Timer` handler
timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
             pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

trainer.run(object_data_loader, EPOCHS)