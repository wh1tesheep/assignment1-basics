import os
import regex as re
from .pretokenization_example import find_chunk_boundaries
from collections import Counter
from collections.abc import Iterable,Iterator
chunk_token = b"<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size:int,
    special_tokens:list[str]
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:

    #初始化词表
    merges: list[tuple[bytes,bytes]] = []
    vocab : dict[int,bytes] = dict()
    for i in range(256):
        vocab[i] = bytes([i])
    special_start = 256
    for i in special_tokens:
        vocab[special_start] = i.encode("utf-8")
        special_start +=1

    #打开文件
    with open(input_path,"rb") as f:
        num_processes = 1 #16 #cpu有16个核心
        boundaries=find_chunk_boundaries(f,num_processes,chunk_token) #将大文件二进制流分块

        #预分词前需要去掉特殊token
        escaped_tokens = [re.escape(token) for token in special_tokens]
        pattern = "|".join(escaped_tokens)

        #多次使用的正则表达式预先编译
        pattern_PAT = re.compile(PAT)

        #开始预分词
        pretoken_bytes:dict[tuple[bytes, ...], int] = {}
        pretoken_counts = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            parts = re.split(pattern, chunk)
            for part in parts:
                pretoken_counts.update(match.group(0) for match in pattern_PAT.finditer(part))
        #print(pretoken_counts) #test
    
        #pretoken_counts仍然是字符串:频数，要转化为pretoken_bytes:dict[tuple[bytes, ...], int]  
        for prestr,counts in pretoken_counts.items():
            #pretoken_bytes[tuple(i.encode("utf-8") for i in prestr)] = counts
            prelist_utf8 = list(i.encode("utf-8") for i in prestr)
            prelist = []
            for i in prelist_utf8:
                for j in i:
                    prelist.append(bytes([j]))
            pretoken_bytes[tuple(prelist)] = counts
        #print(pretoken_bytes) #test
    #关闭文件

    #开始BPE合并

    ##初始字节对统计完毕，开始BPE合并
    while len(vocab) < vocab_size:
        ##首先进行字节对统计，可以直接用pretoken_bytes进行统计，这样规避了special_tokens的边界问题
        pair_counts : dict[tuple[bytes,bytes],int] = Counter()
        for tp,value in pretoken_bytes.items():
            if len(tp) < 2:
                continue
            for i in range(len(tp)-1):
                pair = (tp[i],tp[i+1])
                pair_counts[pair] += value
        pair_max_tuple = max(((pair,frequency) for pair,frequency in pair_counts.items()),key=lambda item:(item[1],item[0]))
        pair_max = pair_max_tuple[0]
        merges.append(pair_max)
        pair_merged = pair_max[0]+pair_max[1]
        vocab[len(vocab)] = pair_merged
        #print(pair_merged)
        ##已经完成一次字节对合并，需要更新pretoken_bytes
        pretoken_bytes_new:dict[tuple[bytes, ...], int] = {} 
        for bytestuple,frequency in pretoken_bytes.items():
            i = 0
            new_tuple = []
            if len(bytestuple) <2:
                pretoken_bytes_new[bytestuple] = frequency
                continue
            while(i<len(bytestuple)-1):
                if((bytestuple[i],bytestuple[i+1])==pair_max):
                    new_tuple.append(pair_merged)
                    i +=2
                else:
                    new_tuple.append(bytestuple[i])
                    i +=1
            if(i == len(bytestuple)-1):
                new_tuple.append(bytestuple[i])
            new_tuple = tuple(new_tuple)
            pretoken_bytes_new[new_tuple] = frequency
        pretoken_bytes = pretoken_bytes_new

    #print(vocab)
    #print(merges)
    #print(pretoken_bytes)
    return (vocab,merges)



    raise NotImplementedError

class Tokenizer:

    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:

        #先开始预分词 用spacial_token分割字符串
        escaped_tokens = [re.escape(token) for token in self.special_tokens]

        raise NotImplementedError
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:


        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:


        raise NotImplementedError
    
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":



        raise NotImplementedError


















if __name__ == "__main__":
    vocab_test,merged_test = train_bpe("data/TinyStoriesV2-GPT4-valid.txt",1000,["<|endoftext|>"])