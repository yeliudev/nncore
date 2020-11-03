# Stage settings
stages = dict(
    epochs=5,
    optimizer=dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-4),
    lr_schedule=dict(type='epoch', policy='step', step=[2, 4]),
    warmup=dict(type='iter', policy='linear', steps=500, ratio=1e-3),
    validation=dict(interval=1))
