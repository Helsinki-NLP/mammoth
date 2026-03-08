"""Define constant values used across the project."""


class DefaultTokens(object):
    PAD = '<pad>'
    BOS = '<s>'
    EOS = '</s>'
    UNK = '<unk>'
    MASK = '<mask>'
    VOCAB_PAD = 'averyunlikelytoken'
    SENT_FULL_STOPS = [".", "?", "!"]
    PHRASE_TABLE_SEPARATOR = '|||'
    ALIGNMENT_SEPARATOR = ' ||| '
    TASK_TOKEN_PREFIX = '<task:'

    @staticmethod
    def task_token(task_name: str) -> str:
        """Build a task-conditioning token string.

        E.g. task_token('summarize') -> '<task:summarize>'
        """
        return f'<task:{task_name}>'


class CorpusName(object):
    VALID = 'valid'
    TRAIN = 'train'
    SAMPLE = 'sample'


class SubwordMarker(object):
    SPACER = '▁'
    JOINER = '￭'
    BEGIN_UPPERCASE = "｟mrk_begin_case_region_U｠"
    END_UPPERCASE = "｟mrk_end_case_region_U｠"
    BEGIN_CASED = "｟mrk_case_modifier_C｠"


class ModelTask(object):
    LANGUAGE_MODEL = 'lm'
    SEQ2SEQ = 'seq2seq'
