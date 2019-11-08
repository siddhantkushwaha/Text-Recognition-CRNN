CHAR_VECTOR = """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_abcdefghijklmnopqrstuvwxyz"""

letters = {ch: i for i, ch in enumerate(CHAR_VECTOR, 0)}
num_classes = len(letters) + 1

max_text_len = 20

img_w = 128
img_h = 64
