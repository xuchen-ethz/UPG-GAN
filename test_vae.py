import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import save_images_vae
from itertools import islice
from util import html
import numpy as np

# helper function
def get_random_z(opt):
    z_samples = np.random.normal(0, 1, (opt.n_samples+1, opt.nz))
    return z_samples
def get_random_z_2(n_samples, nz):
    z_samples = np.random.normal(0, 1, (n_samples, nz))
    return z_samples

# options
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1   # test code only supports batchSize=1
opt.serial_batches = True  # no shuffle

# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch) )
print web_dir
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s' % (opt.name, opt.phase))

# sample random z
if opt.random_walk:
    count = 0
    endpoints = []
    for i, data in enumerate(islice(dataset, opt.how_many)):
        if count > 1:
            break
        count+=1
        model.set_input(data)
        _, real_A, fake_B, real_B, z = model.test_simple(None, encode_real_B=True)
        z = z.cpu().data.numpy()[0]
        endpoints.append(z)
        print z, type(z)

    start = endpoints[0]
    end = endpoints[1]
    delta = (end-start)/opt.n_samples
    z_sample = start
    z_samples = []
    for i in range(opt.n_samples):
        z_sample += delta
        z_samples.append(np.copy(z_sample))
    z_samples.append(end)
    z_samples = np.array(z_samples)
else:
    z_samples = get_random_z(opt)

# test stage
for i, data in enumerate(islice(dataset, opt.how_many)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.how_many))

    for nn in range(opt.n_samples + 1):
        encode_B = nn == 0
        z_sample = z_samples[nn]

        if opt.cond_nc > 0 and not opt.random_walk:
            if encode_B:
                pass
            elif nn < (opt.n_samples + 1)/2:
                z_sample = np.append(z_sample,0)
            else:
                z_sample = np.append(z_sample,1)

        _, real_A, fake_B, real_B, z = model.test_simple(z_sample, encode_real_B=encode_B)
        if nn == 0:
            all_images = [real_A, real_B, fake_B]
            all_names = ['input', 'ground truth', 'encoded']
        else:
            all_images.append(fake_B)
            all_names.append('random sample%2.2d' % nn)

    img_path = 'input image%3.3i' % i
    save_images_vae(webpage, all_images, all_names, img_path, None, width=opt.fineSize)

webpage.save()
