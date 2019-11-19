CHARSET = """ !"#$%&'()*+,-./0123456789:;<=>?@[\]_abcdefghijklmnopqrstuvwxyz{}"""

letters = {ch: i for i, ch in enumerate(CHARSET, 0)}
num_classes = len(letters) + 1

max_text_len = 80

img_w = 128
img_h = 64
