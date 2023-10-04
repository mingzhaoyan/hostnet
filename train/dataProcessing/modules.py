import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
from logging import warning
seed = 68
random.seed(seed)
import sys

sys.path.append('../dataProcessing')

def to_categorical(y, num_classes=None, dtype='float32'):
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


class Loop:
    def __init__(self, data):
        self.sequences = data
        self.num_sequences = len(self.sequences)
        self.seq_length = [len(i) for i in self.sequences]
        self.seq_length.sort()
        self.pad_length = self.seq_length[int(self.num_sequences * 0.95)]

    def pad_self_repeats(self, dtype='int32', split='post', value=0.):
        if not hasattr(self.sequences, '__len__'):
            raise ValueError("sequences must be iterable")
        sample_shape = 0
        for s in self.sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break  

        x = (np.ones((self.num_sequences, self.pad_length) + sample_shape) * value).astype(dtype)

        for idx, s in enumerate(self.sequences):
            if not len(s):
                continue
            elif split == 'post':
                tmp = s[:self.pad_length]
            elif split == 'pre':
                tmp = s[-self.pad_length:]

            tmp = np.asarray(tmp, dtype=dtype)

            repeat_seq = np.array([], dtype=dtype)
            while len(repeat_seq) < self.pad_length:
                # spacer_length = random.randint(1, 50)
                spacer_length = 0
                spacer = [value for _ in range(spacer_length)]
                repeat_seq = np.append(repeat_seq, spacer)
                repeat_seq = np.append(repeat_seq, tmp)

            x[idx, :] = repeat_seq[-self.pad_length:]

        return x

    def run(self):
        return self.pad_self_repeats(), self.pad_length


class Sliding:
    def __init__(self, data):
        self.sequences = data
        self.num_sequences = len(self.sequences)
        self.seq_length = [len(i) for i in self.sequences]

    def straight_output(self, dtype='int32', value=0.):
        if not hasattr(self.sequences, '__len__'):
            raise ValueError("sequences must be iterable")
        sample_shape = 0
        for s in self.sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
        x = (np.ones((self.num_sequences, max(self.seq_length)) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(self.sequences):
            s = np.asarray(s)
            x[idx, :len(s)] = s

        return x

    def run(self):
        return self.straight_output(), self.seq_length


class DealSequence:
    def __init__(self, mode, data, y_encoder, is_x, emb_word_len):
        self.data = data
        self.mode = mode
        self.emb_word_len = emb_word_len
        if mode == 1:
            self.mode = 'loop'
        elif mode == 2:
            self.mode = 'sliding'
        elif mode == 3:
            self.mode = 'no_deal'
        else:
            raise ValueError("choose right mode")

        self.base = 'ATCGN-'
        self.encoder = LabelEncoder()
        self.y_encoder = y_encoder
        self.is_x = is_x   

    def deal_sequence(self):
        out = []
        self.encoder.fit(list(self.base))
        if type(self.data) == str:
            print("str forbidden")
        else:
            for i in self.data:
                out.append(self.encoder.transform(list(i)))

        if self.mode == 'loop':
            out, seq_length = Loop(out).run()
        else:
            out, seq_length = Sliding(out).run()

        # 对长序列进行embedding化
        return self.to_emb(out, self.emb_word_len), seq_length


    def to_emb(self, sequences, word_len):
        relation = {1:"A", 5:"T", 2:"C", 3:"G", 4:"N", 0:"-"}
        res = []  # 词的长度为固定值，位于3-8之间  one-hot的长度为1
        word_len = int(word_len)
        for seq in tqdm(sequences):
            gene = ""
            for i in seq:
                gene += relation[i]
            l = len(gene) - word_len + 1
            tmp = []
            for i in range(l):
                tmp.append(gene[i:i + word_len])
            res.append(tmp)
        return np.array(res)


    def deal_label(self):
        y = self.data
        self.encoder.fit(y)
        if self.y_encoder:
            if np.array(self.encoder.classes_ != self.y_encoder.classes_).all():
                warning(f"Warning not same classes in training and test set")
            useable_classes = set(self.encoder.classes_).intersection(self.y_encoder.classes_)  # 将X和Y放在一起
            try:
                assert np.array(self.encoder.classes_ == self.y_encoder.classes_).all()
            except AssertionError:
                warning(
                    f"not all test classes in training data, only {useable_classes} predictable "
                    f"from {len(self.encoder.classes_)} different classes\ntest set will be filtered so only predictable"
                    f" classes are included")

            try:
                assert len(useable_classes) == len(self.encoder.classes_)  # 判断X和Y的类别长度是否相等
            except AssertionError:
                print("error")
            if not len(useable_classes) == len(self.encoder.classes_):
                global X_test, Y_test
                arr = np.zeros(X_test.shape[0], dtype=int)
                for i in useable_classes:
                    arr[y == i] = 1

                X_test = X_test[arr == 1, :]
                y = y[arr == 1]
                encoded_Y = self.y_encoder.transform(y)
            else:
                encoded_Y = self.encoder.transform(y)

            return to_categorical(encoded_Y, num_classes=len(self.y_encoder.classes_)), self.encoder

        else:
            encoded_Y = self.encoder.transform(y)
            return to_categorical(encoded_Y), self.encoder

    def run(self):
        if self.is_x:
            base = "ATCG" # random replace N
            random.seed(66)
            for i in range(len(self.data)):
                cur = [b for b in self.data[i]]
                for j in range(len(self.data[i])):
                    if cur[j] not in ['A', 'T', 'C', 'G']:
                        cur[j] = base[random.randint(0, 3)]
                self.data[i] = ''.join(cur)
            return self.deal_sequence()
        else:
            return self.deal_label()


class Split:
    def __init__(self, mode, x, y, seq_length, n=35, subseq_length=250):
        self.mode = mode
        self.seq_length = seq_length
        self.x = x
        self.y = y
        self.subseq_length = subseq_length
        self.n = n

    def change_subseq_len(self, value):
        self.subseq_length = value

    def static_split(self):

        batch_size = self.seq_length // self.subseq_length
        print("@@@@@@@@@@@@@@@@@ The number of subsequences each sequence is divided into = ", batch_size)
        newSeqlength = batch_size * self.subseq_length

        bigarray = []
        for sample in tqdm(self.x):
            sample = np.array(sample[0:newSeqlength])
            subarray = sample.reshape((batch_size, self.subseq_length))
            bigarray.append(subarray)
        bigarray = np.array(bigarray)
        x = bigarray.reshape((bigarray.shape[0] * bigarray.shape[1], bigarray.shape[2]))

        y = []
        for i in self.y:
            y.append(batch_size * [i])
        y = np.array(y)
        if len(y.shape) == 2:
            y = y.flatten()
        elif len(y.shape) == 3:
            y = y.reshape((y.shape[0] * y.shape[1], y.shape[2]))

        return x, y, batch_size, [batch_size] * len(self.x)

    def sliding_window(self):
    
        min_seqLen = max(self.n + self.subseq_length, min(self.seq_length))
        print("$$$$$$$$$$$$ The length of the shortest sequence in the current dataset is: ", min_seqLen)

        bigarray = []
        for index, seq in tqdm(enumerate(self.x)):
            seq = list(seq)
            while self.seq_length[index] < min_seqLen:
                self.seq_length[index] *= 2
                seq *= 2

            step = (self.seq_length[index] - self.subseq_length) // self.n
            for i in range(self.n - 1):
                bigarray.append(seq[i * step:self.subseq_length + i * step])
            bigarray.append(seq[-self.subseq_length:])

        x = np.array(bigarray)

        y = []
        for i in self.y:
            y.append(self.n * [i])
        y = np.array(y)
        y = y.reshape((y.shape[0] * y.shape[1], y.shape[2]))

        return x, y, self.n, [self.n] * len(self.x)

    def no_deal(self, _):
        max_seqLen = max(self.seq_length)
        seqs = []
        count = 0
        bigarray = []
        print("generate unprocessed data")

        for index, seq in tqdm(enumerate(self.x)):
            seq = list(seq)
            if self.seq_length[index] < self.subseq_length:
                n = self.subseq_length // self.seq_length[index]
                seq *= n
                res = self.subseq_length - len(seq)
                seq += seq[:res]

                bigarray.append(seq)
                count += 1
                seqs.append(1)
            else:
                nums = self.seq_length[index] // self.subseq_length
                seqs.append(nums)
                for i in range(nums):
                    bigarray[count, :] = seq[i * self.subseq_length: (i+1) * self.subseq_length]
                    count += 1

        x = np.array(bigarray)

        y = []
        for idx, i in enumerate(self.y):
            y.extend([i] * seqs[idx])
        y = np.array(y)

        print("The dimensions of the unprocessed data are as follows: ")
        print(x.shape)
        print(y.shape)

        return x, y, max_seqLen, seqs


    def test_gen(self):
        max_seqLen = max(self.seq_length)
        seqs = []
        count = 0
        bigarray = []
        print("Generate test data")

        for index, seq in tqdm(enumerate(self.x)):
            seq = list(seq)
            if self.seq_length[index] < self.subseq_length:
                while self.seq_length[index] < self.subseq_length:
                    self.seq_length[index] *= 2
                    seq *= 2
                bigarray.append(seq[:self.subseq_length])
                count += 1
                seqs.append(1)
            else:
                nums = self.seq_length[index] // self.subseq_length
                seqs.append(nums)
                for i in range(nums):
                    bigarray.append(seq[i * self.subseq_length: (i + 1) * self.subseq_length])
                    count += 1
        x = np.array(bigarray)

        y = []
        for idx, i in enumerate(self.y):
            y.extend([i] * seqs[idx])
        y = np.array(y)

        print("The dimensions of the test data are as follows: ")
        print(x.shape)
        print(y.shape)

        return x, y, max_seqLen, seqs

    def run(self):
        if self.mode == 1:
            return self.static_split()
        elif self.mode == 2:
            return self.sliding_window()
        elif self.mode == 3:
            return self.no_deal()

