import kerastuner

hp = kerastuner.HyperParameters()

a = hp.Choice('choice', [1, 2, 3])
with hp.conditional_scope('choice', [2, 3]):
    b = hp.Choice('child', [4, 5, 6])
    c = hp.Range('hey', 0, 10)
with hp.conditional_scope('choice', 1):
    b = hp.Choice('child', [4, 5, 6])
    c = hp.Range('hey', 0, 10)
