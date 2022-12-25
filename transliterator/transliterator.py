import time
from typing import List, Tuple, Union

import torch
from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator
import nemo.collections.nlp as nemo_nlp

from utils import split_sentences_into_words, postprocess, unite_batch_hyps_by_words


def chunks(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


class Transliterator:
    def __init__(self, model: str, threshold: float = 0.0, n_hyps: int = 1, batch_size=256) -> None:
        self.nemo_ckpt = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=model).eval()
        if torch.cuda.is_available():
            self.nemo_ckpt = self.nemo_ckpt.cuda()
        self.nemo_ckpt.beam_search = BeamSearchSequenceGenerator(
            embedding=self.nemo_ckpt.decoder.embedding,
            decoder=self.nemo_ckpt.decoder.decoder,
            log_softmax=self.nemo_ckpt.log_softmax,
            bos=self.nemo_ckpt.decoder_tokenizer.bos_id,
            pad=self.nemo_ckpt.decoder_tokenizer.pad_id,
            eos=self.nemo_ckpt.decoder_tokenizer.eos_id,
            max_sequence_length=self.nemo_ckpt.decoder.max_sequence_length,
            beam_size=n_hyps,
            len_pen=0.6,
            max_delta_length=5,
        )
        self.threshold = threshold
        self.n_hyps = n_hyps
        self.batch_size = batch_size

    @staticmethod
    def bring_word_to_transliterator_format(word: str) -> str:
        return " ".join(list(word))

    @staticmethod
    def bring_word_to_human_format(word: str) -> str:
        return "".join(word.split())

    def bring_words_to_transliterator_format(self, words: List[str]) -> List[str]:
        return [self.bring_word_to_transliterator_format(word=word) for word in words]

    def bring_hyps_to_human_format(self, hyps: List[List[Tuple[str, float]]]) -> List[List[Tuple[str, float]]]:
        hyps_in_human_format: List[List[Tuple[str, float]]] = []
        for hyps_word in hyps:
            human_format_hyps: List[Tuple[str, float]] = [
                (self.bring_word_to_human_format(word=word_ru), score) for word_ru, score in hyps_word
            ]
            hyps_in_human_format.append(human_format_hyps)
        return hyps_in_human_format

    def transliterate_batch(self, batch_text: List[str]) -> List[List[Tuple[str, float]]]:
        start_time = time.time()
        best_translits = self.nemo_ckpt.translate(
            text=batch_text,
            source_lang=None,
            target_lang=None,
            return_beam_scores=True,
        )
        batch_hyps = unite_batch_hyps_by_words(
            all_results=best_translits[0], scores=best_translits[1], n_hyps=self.n_hyps
        )
        time_taken = time.time() - start_time
        print(f"Transliterated {len(batch_hyps)} words. It has taken {time_taken:.2f} sec")
        return batch_hyps

    def transliterate_prepared_words(self, src: List[str]) -> List[List[Tuple[str, float]]]:
        batch_text = []
        all_hyps = []
        for batch_text in chunks(lst=src, batch_size=self.batch_size):
            batch_hyps = self.transliterate_batch(batch_text=batch_text)
            all_hyps.extend(batch_hyps)
        return all_hyps

    def transliterate(self, src: List[str]) -> List[Union[str, Tuple[str, float]]]:
        human_format_words, nums_of_words = split_sentences_into_words(src=src)
        transliterator_format_words = self.bring_words_to_transliterator_format(words=human_format_words)
        hyps = self.transliterate_prepared_words(src=transliterator_format_words)
        hyps = self.bring_hyps_to_human_format(hyps=hyps)
        return postprocess(
            words_en=human_format_words, hyps=hyps, nums_of_words=nums_of_words, threshold=self.threshold
        )
