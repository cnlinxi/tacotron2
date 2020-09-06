import os


def format_biaobei_to_ijspeech(input_path, output_path):
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            index, text = line.split('	')
            text = text.replace('#1', '').replace('#2', '').replace('#3', '').replace('#4', '')
            if not index.isdigit():
                continue
            fout.write('{}|{}\n'.format(index, text).encode('utf-8'))


def extract_pure_text(input_path, output_path):
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            index, text = line.split('|')
            fout.write('{}\n'.format(text).encode('utf-8'))


def add_index(input_path, output_path):
    index = 1
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            line = '{:06d}|{}\n'.format(index, line)
            fout.write(line.encode('utf-8'))
            index += 1


if __name__ == '__main__':
    base_dir = 'data'
    input_path = os.path.join(base_dir, '000001-010000.txt')
    output_path = os.path.join(base_dir, 'biaobei.meta')
    output_pure_text_path = os.path.join(base_dir, 'biaobei.puretext')
    # extract_pure_text(output_path, output_pure_text_path)
    # format_biaobei_to_ijspeech(input_path, output_path)
    input_pinyin_path = os.path.join(base_dir, 'biaobei.pinyin')
    add_index(input_pinyin_path, output_path)
