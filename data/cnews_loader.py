# coding: utf-8
import os
import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import random

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='gbk', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        flag = ''
        for line in f:
            try:
                if line != None:
                    if len(line) <12:
                        flag = line;
                    else:
                        label = flag
                        content = line.strip().split(' ')
                    if content:
                        contents.extend(list(native_content(content)))
                        i = 1
                        while i <= len(content):
                            labels.append(native_content(label.strip()))
                            i = i + 1
                else:
                    break
            except:
                pass
        comb = list(zip(contents,labels))
        random.shuffle(comb)
        contents[:],labels[:] = zip(*comb)   #将contents和labels按相同顺序打乱
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=230):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    open_file(vocab_dir, mode='w').write('\n'.join(words))


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['分数线', '食堂', '宿舍', '官网', '英文名', '专业', '学院', '收费', '地址', '邮编', '占地', '邮箱', '招办电话', '学校性质', '硕士点', '博士点', '校庆日', '知名校友', '就业情况', '创办时间', '学校代码', '师资力量', '学术资源', '科研成果']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

# base_dir = os.path.dirname(os.path.abspath('__file__'))
# train_dir = os.path.join(base_dir, 'train.txt')
# test_dir = os.path.join(base_dir, 'test.txt')
# vocab_dir = os.path.join(base_dir, 'vocab.txt')
#
