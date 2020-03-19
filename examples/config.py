stages = [
    dict(
        epochs=2,
        val_interval=1,
        optimizer=dict(type='SGD', lr=1e-3, momentum=0.9),
        grad_clip=None,
        loss='loss')
]

hooks = [
    dict(type='IterTimerHook'),
    dict(type='OptimizerHook'),
    dict(
        type='LoggerHook',
        interval=100,
        writers=[dict(type='CommandLineWriter'),
                 dict(type='JSONWriter')])
]

work_dir = 'work_dirs/mnist'
