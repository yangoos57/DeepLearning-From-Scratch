from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm


class IMDBBertData(Dataset):
    # special token
    CLS = "[CLS]"  # 문장 시작
    PAD = "[PAD]"  # 빈공간
    SEP = "[SEP]"  # 문장 끝
    MASK = "[MASK]"
    UNK = "[UNK]"

    MASKED_INDICES_COLUMN = "masked_indices"
    TARGET_COLUMN = "indices"

    # nsp 용도
    NSP_TARGET_COLUMN = "is_next"
    TOKEN_MASK_COLUMN = "token_mask"

    mask_percent = 0.15

    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(
        self, path, ds_from=None, ds_to=None, should_include_text=False
    ) -> None:
        """
        should_include_text = True : Debug 모드
        """
        # load dataset
        self.ds = pd.read_csv(path)["review"]

        # slice dataset
        if ds_from is not None or ds_to is not None:
            self.ds[ds_from:ds_to]

        self.tokenizer = get_tokenizer("basic_english")
        self.counter = Counter()
        self.vocab = None

        self.optimal_sentence_length = None
        self.should_include_text = should_include_text

        # self.df(dataframe) 만들 때 column 정의
        if should_include_text:
            self.columns = [
                "masked_sentence",
                self.MASKED_INDICES_COLUMN,
                "sentence",
                self.TARGET_COLUMN,
                self.TOKEN_MASK_COLUMN,
                self.NSP_TARGET_COLUMN,
            ]

        else:
            self.columns = [
                self.MASKED_INDICES_COLUMN,
                self.TARGET_COLUMN,
                self.TOKEN_MASK_COLUMN,
                self.NSP_TARGET_COLUMN,
            ]

        # 최종 값 df에 저장
        self.df = self.prepare_dataset()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        getitem은 class에서 바로 슬라이싱을 하면 실행하는 함수임
        ex) IMDBBertData[0:1] => return (inp,attention_mask ...)
        현재는 getitem을 활용해 trainin을 간편하게 하려는 목적임
        """

        item = self.df.iloc[index]
        inp = torch.Tensor(item[self.MASKED_INDICES_COLUMN]).long()  # int64 타입지정
        token_mask = torch.Tensor(item[self.TOKEN_MASK_COLUMN]).bool()

        mask_target = torch.Tensor(item[self.TARGET_COLUMN]).long()  # int64 타입지정
        mask_target = mask_target.masked_fill_(token_mask, 0)

        # 훈련 시 [PAD] 제거용으로 쓰인다고 함.
        attention_mask = (inp == self.vocab[self.PAD]).unsqueeze(0)

        if item[self.NSP_TARGET_COLUMN] == 0:
            t = [1, 0]
        else:
            t = [0, 1]
        nsp_target = torch.Tensor(t)
        return (inp, attention_mask, token_mask, mask_target, nsp_target)

    def prepare_dataset(self):
        """
        Main Function
        Bert에 활용 될 Dataset 만드는 function
        """

        # vocab에 단어 저장
        # vocab 사용법 : vocab(['here','is','the','example']) => to indices
        # vocab.lookup_indices()
        # vocab.lookup_token()

        sentences = []
        nsp = []
        sentence_lens = []

        # For MLM
        for review in self.ds:
            review_sentences = review.split(". ")
            sentences += review_sentences  # extend 대신 + 사용해도 됨
            self._update_length(review_sentences, sentence_lens)

        self.optimal_sentence_length = self._find_optimal_sentence_length(sentence_lens)

        print("vocab 생성")
        for sentence in tqdm(sentences):
            s = self.tokenizer(sentence)
            self.counter.update(s)

        self._fill_vocab()

        # For NSP
        print("데이터 전처리 시작")
        for review in tqdm(self.ds):
            review_sentences = review.split(". ")
            if len(review_sentences) > 1:
                for i in range(len(review_sentences) - 1):
                    # True NSP item
                    first, second = self.tokenizer(review_sentences[i]), self.tokenizer(
                        review_sentences[i + 1]
                    )
                    nsp.append(self._create_item(first, second, 1))
                    # False NSP item
                    first, second = self._select_false_nsp_sentences(sentences)
                    first, second = self.tokenizer(first), self.tokenizer(second)
                    nsp.append(self._create_item(first, second, 0))

        df = pd.DataFrame(nsp, columns=self.columns)
        return df

    # For MLM
    def _update_length(self, input: list, output):
        """
        개별 문장의 단어 개수 반환
        """
        for sen in input:
            output.append(len(sen.split(" ")))

    def _find_optimal_sentence_length(self, lengths: list):
        """
        lengths = dataset에 있는 문장 길이
        return : 70% 범위의 문장 길이
        IMDB는 27개라고 함.
        """
        arr = np.array(lengths)
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

    def _fill_vocab(self):
        # 2번 이상 반복되는 단어만 저장하기
        self.vocab = vocab(self.counter, min_freq=2)

        self.vocab.insert_token(self.CLS, 0)
        self.vocab.insert_token(self.PAD, 1)
        self.vocab.insert_token(self.MASK, 2)
        self.vocab.insert_token(self.SEP, 3)
        self.vocab.insert_token(self.UNK, 4)
        self.vocab.set_default_index(4)

    def _create_item(self, first: list, second: list, target: int = 1):
        """
        NSP 훈련용으로 활용
        1. Masked Sentence 생성
        2. random words Sentence 생성
        """

        # 1.masked Sentence 생성
        updated_first, first_mask = self._preprocess_sentence(first.copy())
        updated_second, second_mask = self._preprocess_sentence(second.copy())

        nsp_sentence = updated_first + [self.SEP] + updated_second
        nsp_indicies = self.vocab.lookup_indices(nsp_sentence)
        inverse_token_mask = first_mask + [True] + second_mask

        # 2.masked Sentence 생성
        first, _ = self._preprocess_sentence(first.copy(), should_mask=False)
        second, _ = self._preprocess_sentence(second.copy(), should_mask=False)
        original_nsp_sentence = first + [self.SEP] + second
        original_nsp_indices = self.vocab.lookup_indices(original_nsp_sentence)

        if self.should_include_text:
            return (
                nsp_sentence,
                nsp_indicies,
                original_nsp_sentence,
                original_nsp_indices,
                inverse_token_mask,
                target,
            )
        else:
            return (nsp_indicies, original_nsp_indices, inverse_token_mask, target)

    # For NSP Sentence
    def _select_false_nsp_sentences(self, sentences: list):
        """
        false NSP 만들 sentence 선택하기
        sentences : 문장 리스트 전체
        return : 임의로 문장 2개 선택(앞과 뒤는 연결되지 않음)
        """
        sentences_len = len(sentences)
        sentence_index = random.randint(0, sentences_len - 1)
        next_sentence_index = random.randint(0, sentences_len - 1)

        while next_sentence_index == sentence_index + 1:
            next_sentence_index = random.randint(0, sentences_len - 1)

        return sentences[sentence_index], sentences[next_sentence_index]

    def _preprocess_sentence(self, sentence: list, should_mask: bool = True):
        """
        mask 퍼센트(15%)를 [mask]로 바꾸는 매소드
        sentences : 문장 리스트 전체
        return : mask 된 문장, inverse token mask
        """
        inverse_token_mask = None
        if should_mask:
            sentence, inverse_token_mask = self._mask_sentence(sentence)
        sentence, inverse_token_mask = self._pad_sentence(
            [self.CLS] + sentence, inverse_token_mask
        )

        return sentence, inverse_token_mask

    def _mask_sentence(self, sentence: list):
        """
        mask 퍼센트(15%)를 [mask] 또는 랜덤한 단어로 바뀜
        이때 [mask] : random_word = 8 : 2 비율
        sentence : 문장 하나
        return : mask 또는 random 단어 변환 된 문장, inverse token mask
        """
        len_s = len(sentence)
        # len(len_s, 27)
        # masked 된 문장 또는 단어가 바뀐 경우 False로 변환됨
        inverse_token_mask = [
            True for _ in range(max(len_s, self.optimal_sentence_length))
        ]

        mask_amount = round(len_s * self.mask_percent)

        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)

            if random.random() < 0.8:
                # mask_amount의 80%는 mask로
                sentence[i] = self.MASK
            else:
                # # mask_amount의 20%는 단어 바꾸기
                # 5인 이유는 0,1,2,3,4 토큰이 모두 special token이기 때문
                j = random.randint(5, len(self.vocab) - 1)

                # 단어 바꾸기
                sentence[i] = self.vocab.lookup_token(j)

            inverse_token_mask[i] = False

        return sentence, inverse_token_mask

    def _pad_sentence(self, sentence: list, inverse_token_mask: bool):
        len_s = len(sentence)

        # self.optimal_sentence_length = 27단어
        if len_s >= self.optimal_sentence_length:
            s = sentence[: self.optimal_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - len_s)

        if inverse_token_mask:
            len_m = len(inverse_token_mask)
            if len_m >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[: self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (
                    self.optimal_sentence_length - len_m
                )

        return s, inverse_token_mask


a = IMDBBertData("./data/IMDB Dataset.csv", should_include_text=True)

a.vocab(["here", "is", "the", "example"])
