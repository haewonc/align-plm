'''
Hard-coded for specific resource constraints 
Only consider Large/Small checkpoints
'''

def get_batch_size_align(length, size):
    if length < 100:
        batch_size = 10
    elif length < 200:
        batch_size = 6
    elif length < 300:
        batch_size = 4
    elif length < 360:
        batch_size = 3
    elif length < 450:
        batch_size = 2
    else:
        batch_size = 1
    if size == 'Tranception_Small':
        batch_size = batch_size * 6
    return batch_size 

def get_batch_size(length, size):
    if length < 250:
        batch_size = 12
    elif length < 400:
        batch_size = 6
    elif length < 512:
        batch_size = 3
    else:
        batch_size = 1
    if size == 'Tranception_Small':
        batch_size = batch_size * 6
    return batch_size 

def get_inference_batch_size(length, size):
    if length < 250:
        batch_size = 48
    elif length < 400:
        batch_size = 24
    elif length < 512:
        batch_size = 12
    else:
        batch_size = 4
    if size == 'Tranception_Small':
        batch_size = batch_size * 6
    return batch_size 