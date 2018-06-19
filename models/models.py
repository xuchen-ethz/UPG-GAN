
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'ae_cycle_gan_all':
        assert (opt.dataset_mode in ['unaligned','unaligned_with_label'])
        from .ae_cycle_gan_model import AECycleGANModel
        model = AECycleGANModel()
    elif opt.model == 'vae_cycle_gan':
        assert (opt.dataset_mode in ['unaligned','unaligned_with_label'])
        from .vae_cycle_gan_model import VAECycleGANModel
        model = VAECycleGANModel()
    elif opt.model == 'ultimate':
        assert (opt.dataset_mode in ['unaligned','unaligned_with_label'])
        from .ultimate_model import VAECycleGANModelAll
        model = VAECycleGANModelAll()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
