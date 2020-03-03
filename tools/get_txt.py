import os

path = r'../dataSet/Images/picture'
start = 30


def get_filename(self):
    """
    得到文件名
    适用于一个文件夹下所有图片已经标号有序的情况
    """
    with open(os.path.join(self.output, 'trainval.txt'), 'a+') as f:
        for file in sorted(os.listdir(self.path)):
            name = file.strip().split('.')[0]
            f.write(str(name) + '\n')


def rename_and_get_filename():
    """
    重命名以及得到文件名
    适用于每个文件夹归属一个类别的情况
    如: a文件夹下, 全为a的图片
    """
    # 得到path下的所有文件夹(排序)
    files = [os.path.join(path, x) for x in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, x))]
    for folder in files:
        # Got image's class
        splits = folder.split('/')
        class_name = splits[len(splits) - 1]
        print('Processing Class %s ...' % class_name)
        # Init counter
        count = start
        for file in sorted(os.listdir(folder)):
            # Got image's suffix
            suffix = str(file.split('.')[1])
            old = os.path.join(folder, file)
            new = os.path.join(folder, (str(class_name) + '_' + str(count) + '.' + suffix))
            # rename
            os.rename(old, new)
            count += 1
    print('Done!')


def check_files():
    """
    检查重命名后的文件是否丢失
    """
    # 得到path下的所有文件夹(排序)
    files = [os.path.join(path, x) for x in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, x))]
    for folder in files:
        cnt = 1
        for file in os.listdir(folder):
            cnt += 1
        print('The Summary of %s is : %s' % (folder, cnt))


if __name__ == '__main__':
    rename_and_get_filename()
    check_files()
