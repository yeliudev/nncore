stages = [
    dict(
        epochs=5,
        val_interval=1,
        optimizer=dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-4),
        lr_schedule=dict(type='epoch', policy='step', step=[2, 4]),
        warmup=dict(type='iter', policy='linear', steps=500, ratio=1 / 3.0))
]

hooks = [
    dict(type='IterTimerHook'),
    dict(type='LrUpdaterHook'),
    dict(type='WarmupHook'),
    dict(type='OptimizerHook'),
    dict(type='CheckpointHook'),
    dict(
        type='EventWriterHook',
        interval=50,
        writers=[
            dict(type='CommandLineWriter'),
            dict(type='JSONWriter'),
            dict(type='TensorboardWriter')
        ])
]

work_dir = 'work_dirs/mnist'
