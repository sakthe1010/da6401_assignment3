from dataset import read_data

TRAIN_FILE = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv"
pairs = read_data(TRAIN_FILE)

# compute lengths (including <sos> and <eos>)
lengths = [len(src) + 2 for src, _ in pairs]  # +2 for <sos> & <eos>
print("Max source length:", max(lengths))
print("Mean source length:", sum(lengths)/len(lengths))
print("95th percentile:", sorted(lengths)[int(0.95*len(lengths))])
