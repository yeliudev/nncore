stages = [
    dict(
        epochs=5,
        val_interval=1,
        optimizer=dict(type='SGD', lr=1e-2, momentum=0.9),
        lr_updater=dict(
            policy='step',
            type='epoch',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=1.0 / 3,
            step=[3, 5]),
        grad_clip=None,
        loss='loss')
]

hooks = [
    dict(type='IterTimerHook'),
    dict(type='LrUpdaterHook'),
    dict(type='OptimizerHook'),
    dict(type='CheckpointHook'),
    dict(
        type='EventWriterHook',
        interval=50,
        writers=[dict(type='CommandLineWriter'),
                 dict(type='JSONWriter')])
]

work_dir = 'work_dirs/mnist'
