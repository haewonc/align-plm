def get_context(mutated_s, mutated_e, length):
    if length > 768:
        trunc_len = mutated_e - mutated_s -1
        if trunc_len > 768:
            pad_size = 0
        else:
            pad_size = (768-trunc_len)//2
        mutated_s, mutated_e = max(0, mutated_s-pad_size), min(mutated_e+pad_size, length)
        truncated = True
    else:
        mutated_s, mutated_e = 0, length
        truncated = False
    return mutated_s, mutated_e, truncated