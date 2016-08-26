import os
import random

data_dir = 'data'
depth = 2
height = 256
width = 256

def get_filename_set(data_set):
    labels = []
    filename_set = []

    with open(data_dir + '/meta/labels.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            labels += inner_list

    for i, lable in enumerate(labels):
        list = os.listdir(data_dir  + '/' + data_set + '/' + lable)
        for filename in list:
            filename_set.append([i, data_dir  + '/' + data_set + '/' + lable + '/' + filename])

    random.shuffle(filename_set)
    return filename_set

def convert_images(data_set):
    filename_set = get_filename_set(data_set)

    for i in range(0, len(filename_set)):
        print('############### %d %d ###############' % (i, filename_set[i][0]))
        os.system('echo %d >> %s/%s.bin' % (filename_set[i][0], data_dir, data_set))
        os.system('ffmpeg -i %s -vf scale=%d:%d out.yuv;cat out.yuv >> %s/%s.bin;rm -f out.yuv' % (filename_set[i][1], height, width, data_dir, data_set))

def main(argv = None):
    convert_images('train')

if __name__ == '__main__':
    main()
