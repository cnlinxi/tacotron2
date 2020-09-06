# -*- coding: utf-8 -*-
# @Time    : 2018/12/10 20:22
# @Author  : MengnanChen
# @FileName: transform_pinyin.py
# @Software: PyCharm

import os
import re
import codecs


class words2pinyin:
    def __init__(self, path_pinyin=None, path_text=None):
        self.path_pinyin = path_pinyin
        self.path_text = path_text

    def load_pinyinDict(self):
        dic = {}
        with codecs.open(self.path_pinyin, 'r', encoding='utf8') as f1:
            while True:
                line = f1.readline()
                if len(line) == 0:
                    break
                word, pinyin = line.strip().split('=')
                dic[word] = pinyin.split(',')
        return dic

    def load_text(self):
        '''
        :return:返回以逗号隔开的文本list ，形如[[...],[...],...]
        '''
        chars = []
        with codecs.open(self.path_text, 'r', encoding='utf8') as f2:
            for line in f2.readlines():
                chars.append(line.strip().split())
        return chars

    # 之后有空研究，可以改成python版AC自动机算法来多模式匹配。
    def transform(self, text_chars=None, text_file_list=None, use_space=False):
        '''
        :param text_chars: 能自动识别list还是str，但多行文本要使用text_file_list
        :param text_file_list: 多行文本传入，必须要是分词后并用逗号隔开的list格式传入
        :param use_space: 结果用空格隔开
        :return:文本的拼音list
        '''
        self.use_space = use_space
        pinyin_dic = self.load_pinyinDict()
        if type(text_chars) is str:
            self.text_chars = [text_chars.split(' ')]
        if type(text_chars) is list:
            self.text_chars = [text_chars]
        if text_file_list:
            # 必须是分好词的文本，并以逗号隔开
            self.text_chars = text_file_list
        if self.path_text:
            # 必须要是分词好的文本，并把以逗号隔开的list传入
            self.text_chars = self.load_text()
        # 正向匹配最大分词算法
        # 遍历拼音词典的keys，获得最大词长度（也可理解为最大拼音长度）
        max_pinyin = 0
        for key in pinyin_dic.keys():
            if len(key) > max_pinyin:
                max_pinyin = len(key)
        # 空列表存储多行文本的拼音结果
        self.words = []
        # text_chars是分好词的文本，len(text_chars)文本的长度
        for j in range(len(self.text_chars)):
            # j表示每句话
            word = []
            for chars in self.text_chars[j]:
                # chars表示 分词里面的每个分词
                # 临时存储的分词拼音
                word_ = []
                n = 0
                while n < len(chars):
                    matched = 0
                    for i in range(max_pinyin, 0, -1):
                        s = chars[n: n + i]  # 截取文本字符串n到n+i位,最后到n+1位
                        # 判断所截取字符串是否在分词词典和停用词词典内
                        if s in pinyin_dic:
                            if len(s) == 1:
                                # 单字默认提取第一个拼音
                                pinyin_str = pinyin_dic[s][0]
                                # word当前最后一个词是拼音而且总长度大于0
                                if len(word_) > 0:
                                    if chars[n - 1] == '一' and pinyin_str[-1] == str(4):
                                        if not s in '二四六':
                                            word_[-1] = word_[-1][:-1] + str(2)
                                    if chars[n - 1] == '一' and pinyin_str[-1] != str(4):
                                        if not s in '三五七八九十':
                                            word_[-1] = word_[-1][:-1] + str(4)
                                    if chars[n - 1] == '不' and pinyin_str[-1] == str(4):
                                        word_[-1] = word_[-1][:-1] + str(2)
                                    if chars[n - 1] == '不' and pinyin_str[-1] == str(4):
                                        word_[-1] = word_[-1][:-1] + str(4)
                                word_.append(pinyin_str)
                            else:
                                word_.extend(pinyin_dic[s])
                            matched = 1
                            n = n + i
                            break
                    # 判断标志位是否匹配，不匹配则直接将当前字符输出，然后字位 + 1
                    if not matched:
                        word.append(chars[n])
                        n = n + 1
                if len(word_) > 1:
                    new_word = word_
                    for j in range(len(new_word) - 1, 0, -1):
                        i = j - 1
                        if new_word[j][-1] == str(3) and new_word[i][-1] == str(3):
                            new_word[i] = new_word[i][:-1] + str(2)
                    word_ = new_word
                word.extend(word_)
            if self.use_space:
                self.words.append(' '.join(word))
            else:
                self.words.append(word)
        return self.words

    def save_result(self, path_save):
        pinyin = self.words
        text = self.text_chars
        assert len(pinyin) == len(text)
        with open(path_save, 'w', encoding='utf8') as f3:
            if self.use_space:
                for i, word in enumerate(pinyin):
                    f3.write(str("%08d" % i) + '  ' + pinyin[i] + '\n')
            else:
                for j, word_list in enumerate(pinyin):
                    f3.write(str("%08d" % j) + '  ' + ' '.join(pinyin[j]) + '\n')


class TacotronPreprocess:
    def __init__(self, pinyin_dict_path=os.path.join(os.path.dirname(__file__), 'dependency_data/pinyin.dict')):
        # keep symbols in clean_symbols(), some symbols such as '、','；' will be replaced by '，'
        self._keep_symbols = ['？', '！', '。', '，', '、', '；']
        self._word_boundary = '&'  # mark word boundary with '&'
        self._pinyin_boundary = ' '  # the requirement mark by pinyin
        self._prosody_mark = ['#', '*', '$', '%']
        _wb = self._word_boundary
        _pb = self._pinyin_boundary
        _pm1, _pm3 = self._prosody_mark[0], self._prosody_mark[2]
        self._replacement_expression = [(re.compile('{}'.format(x[0]), re.IGNORECASE), x[1]) for x in [
            ('{}，{}'.format(_wb, _wb), '{}{}{}，{}'.format(_pb, _pm3, _pb, _pb)),
            ('{}。{}'.format(_wb, _wb), '{}{}{}。{}'.format(_pb, _pm3, _pb, _pb)),
            ('{}！{}'.format(_wb, _wb), '{}{}{}！{}'.format(_pb, _pm3, _pb, _pb)),
            ('{}？{}'.format(_wb, _wb), '{}{}{}？{}'.format(_pb, _pm3, _pb, _pb)),
            ('{}、{}'.format(_wb, _wb), '{}{}{}，{}'.format(_pb, _pm3, _pb, _pb)),
            ('{}；{}'.format(_wb, _wb), '{}{}{}，{}'.format(_pb, _pm3, _pb, _pb)),
            (self._word_boundary, '{}{}{}'.format(_pb, _pm1, _pb)),
        ]]
        self._number_expression = [(re.compile('{}'.format(x[0]), re.IGNORECASE), x[1]) for x in [
            ('(?P<value>\d+年)', self.number2chinese_utils),  # 1996年
            ('(?P<value>\$\d+(\.\d+))', self.number2chinese_utils),  # $12.5
            ('(?P<value>\￥\d+(\.\d+))', self.number2chinese_utils),  # ￥12.5
            ('(?P<value>\d+(\.\d+)%)', self.number2chinese_utils),  # 12.5%
            ('(?P<value>\d+(\.\d+))', self.number2chinese_utils),  # 123.4
            ('(?P<value>\d+/\d+)', self.number2chinese_utils),  # 10/13
            ('(?P<value>\d+)', self.number2chinese_utils),  # 123
            ('W', '万'),
            ('w', '万'),
            ('K', '千'),
            ('k', '千'),
            ('-', '至'),
            ('~', '至'),
            ('/', '每'),
            ('°', '度'),
            ('’', '分'),
            ('<', '小于'),
            ('\+', '加'),
        ]]
        self._pinyin_dict_path = pinyin_dict_path

    def _is_chinese_char(self, cp):
        # check cp is chinese or not
        cp = ord(cp)
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
                (cp >= 0x3400 and cp <= 0x4DBF) or
                (cp >= 0x20000 and cp <= 0x2A6DF) or
                (cp >= 0x2A700 and cp <= 0x2B73F) or
                (cp >= 0x2B740 and cp <= 0x2B81F) or
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or
                (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True

        return False

    def _e2c_punctuation(self, text):
        # translate english punctuation to chinese punctuation
        e_pun = u',.!?[]()<>"\''
        c_pun = u'，。！？【】（）《》“‘'
        table = {ord(f): ord(t) for f, t in zip(e_pun, c_pun)}
        return text.translate(table)

    def clean_symbols(self, input_path, output_path):
        # clean symbol in text, allow symbols in self._keep_symbols and '、','；' etc will be replaced by ','
        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            for line in fin:
                line = line.decode('utf-8').strip('\r\n ')
                line = self._e2c_punctuation(line)  # translate english symbols to chinese symbols
                line_t = []
                for char in line:
                    if self._is_chinese_char(char) \
                            or char in self._keep_symbols \
                            or char.isdigit() \
                            or char.isalnum() \
                            or char == ' ':
                        line_t.append(char)
                line = '{}\n'.format(''.join(line_t))
                fout.write(line.encode('utf-8'))

    def _independence_number2chinese(self, number: str):
        # translate arabic to independent chinese
        number = number.strip('\r\n ')
        if len(number) < 1 or not number.isdigit():
            return ''
        digit_to_chi = {
            '0': '零',
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九',
        }
        result = [digit_to_chi[n] for n in number]
        return ''.join(result)

    def _integer_number2chinese(self, number: str):
        # translate intergral number into chinese, only support to 999,9999(max: 999,9999,9999)
        number = number.strip('\r\n ')
        try:
            number = int(number)
        except:
            return ''
        if not isinstance(number, (int,)) or number < 0:
            return ''
        elif number >= pow(10, 7):
            return self._independence_number2chinese(str(number))
        if number == 0:
            return '零'

        digit_to_hanzi = {
            '0': '零',
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九',
        }
        # for every 4 position
        sections = {
            4: '万',
            8: '亿',
        }
        # the number2chinese for 1~999, for example, 350 -> 3百, 5十
        per_section = {
            1: '十',
            2: '百',
            3: '千',
        }
        # number_string = unicode(number)
        number_string = str(number)

        def convert_four_number(num):
            # num is already inverse order
            res = ''
            for i, val in enumerate(num):
                tmp = digit_to_hanzi.get(val)
                if val != '0':
                    tmp += per_section.get(i % 4, '')
                res = tmp + res
            return res

        result = []
        four_number_round = ''
        total_count = len(number_string)
        for i, val in enumerate(reversed(number_string)):
            if i in sections:
                result.insert(0, sections[i])
            # for every 4 position
            four_number_round += val
            if len(four_number_round) < 4 and i + 1 < total_count:
                continue

            insert_val = convert_four_number(four_number_round)
            # put last 4 position directly
            if i < 4:
                insert_val = insert_val.rstrip('零')
            # all zeros, replace '万', '亿' with '零'
            elif all([i == '零' for i in insert_val]):
                result[0] = '零'
            # non-all-zeros
            # Then: multiple zeros at the end are merged into one zero and swap the position with '万'/'亿'
            elif insert_val.endswith('零'):
                insert_val = insert_val.rstrip('零')
                pos_zero = result.pop(0)
                insert_val = insert_val + pos_zero + '零'

            result.insert(0, insert_val)
            four_number_round = ''
        result = ''.join(result)

        # 10: 一十 -> 十
        # 10,0000: 一十万 -> 十万
        if result.startswith('一十'):
            result = result.lstrip('一')

        # del redundant '零'
        result = result.strip('零')
        # 1001: 一千零零一 -> 一千零一
        result = re.sub(r'零+', '零', result)

        return result

    def _float_number2chinese(self, number: str):
        number = number.strip('\r\n ')
        int_number = number.split('.')[0]
        float_number = ''.join(number.split('.')[1:])
        result = '{}点{}'.format(self._integer_number2chinese(int_number),
                                self._independence_number2chinese(float_number))
        return result

    def number2chinese_utils(self, number):
        # number: re object
        try:
            number = str(number.group('value'))
        except:
            print('number error in number2chinese_utils, get object: {}'.format(number))

        if '年' == number[-1]:  # '1997年'
            result = '{}年'.format(self._independence_number2chinese(number[:-1]))
        elif '$' == number[0]:  # '$12.5'
            result = '{}美元'.format(self._float_number2chinese(number[1:]))
        elif '￥' == number[0]:  # '￥12.5'
            result = '{}元'.format(self._float_number2chinese(number[1:]))
        elif '%' == number[-1]:  # '12.5%'
            result = '百分之{}'.format(self._float_number2chinese(number[:-1]))
        elif '/' in number:  # '10/13'
            son = number.split('/')[0]
            father = ''.join(number.split('/')[1:])
            result = '{}分之{}'.format(self._integer_number2chinese(father), self._integer_number2chinese(son))
        elif '.' in number:  # '12.5'
            result = self._float_number2chinese(number)
        else:
            result = self._integer_number2chinese(number)

        return result

    def number2chinese(self, input_path, output_path):
        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            for line in fin:
                line = line.decode('utf-8').strip('\r\n ')
                for regex, replacement in self._number_expression:
                    line = re.sub(regex, replacement, line)
                line = '{}\n'.format(line)
                fout.write(line.encode('utf-8'))

    def prosody_prediction(self, input_path, output_path):
        # word boudary will be '#1'
        # '，' etc  punctuation will be '#3'
        # the end of sentences will be '#4'
        wordcut_type = None
        try:
            from pyhanlp import HanLP
            wordcut_type = 'hanlp'
        except ImportError:
            print('cannot import HanLP')

        if wordcut_type is None:
            try:
                import jieba
                wordcut_type = 'jieba'
            except ImportError:
                print('cannot import jieba')

        assert wordcut_type is not None, 'no package to word cut'

        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            for line in fin:
                line = line.decode('utf-8').strip('\r\n ')

                if wordcut_type == 'hanlp':
                    line = HanLP.segment(line)
                    line = [str(x).split('/')[0] for x in line]
                    line = self._word_boundary.join(line)
                elif wordcut_type == 'jieba':
                    # add new word here
                    jieba.suggest_freq('您的', tune=True)
                    jieba.suggest_freq('您呢', tune=True)

                    line = self._word_boundary.join(jieba.cut(line, HMM=True))

                for regex, replacement in self._replacement_expression:
                    line = re.sub(regex, replacement, line)
                try:
                    if line[-1] not in self._keep_symbols:
                        line = '{} {} {}\n'.format(line[:-4], self._prosody_mark[3],
                                                   '。')  # default end of utterence is '。'
                    else:
                        # input line: '我行 # 邀请 # 你 # 办理 # 业务 # 。'
                        # output line: '我行 # 邀请 # 你 # 办理 # 业务 % 。'
                        line = '{} {} {}\n'.format(line[:-4], self._prosody_mark[3], line[-1])
                except:
                    print('error in wordcut, line:', line)
                fout.write(line.encode('utf-8'))

    def transform_pinyin(self, input_path, output_path):
        # transform chinese to pinyin
        # text format should be ''
        # which means prosody mark should by gapped by self._pinyin_boundary
        assert os.path.isfile(self._pinyin_dict_path), 'pinyin dict not found'
        self._w2p = words2pinyin(path_pinyin=self._pinyin_dict_path, path_text=input_path)
        res = self._w2p.transform()
        res = ['{}\n'.format(' '.join(x)).encode('utf-8') for x in res]
        with open(output_path, 'wb') as fout:
            fout.writelines(res)

    def text2pinyin(self, input_path, output_path):
        # directly transform pure text file to pinyin file
        # include: number to chinese, clean symbols, prosody prediction, transform pinyin
        self._tmp_dir = 'tmp_data'
        os.makedirs(self._tmp_dir, exist_ok=True)
        tmp_output_path1 = os.path.join(self._tmp_dir, 'tmp_data1')
        tmp_output_path2 = os.path.join(self._tmp_dir, 'tmp_data2')
        tmp_output_path3 = os.path.join(self._tmp_dir, 'tmp_data3')

        # first number to chinese then clean symbols, because 12.5, 12.5%, 12元/股 etc.
        self.number2chinese(input_path, tmp_output_path1)
        self.clean_symbols(tmp_output_path1, tmp_output_path2)
        self.prosody_prediction(tmp_output_path2, tmp_output_path3)
        self.transform_pinyin(tmp_output_path3, output_path)


if __name__ == '__main__':
    test_input_path = 'data/biaobei.puretext'
    test_output_path = 'data/biaobei.pinyin'
    taco_pre = TacotronPreprocess()
    taco_pre.text2pinyin(test_input_path, test_output_path)
