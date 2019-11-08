from data_generator import DataGenerator

gen = DataGenerator(data_path='../FUNSD_TEXT_RECOGNITION/train_data/', batch_size=16)

for i in range(5):
    o = gen.__getitem__(i)
    print(o[0][1])
