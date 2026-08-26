import regex as re
from .pretokenization_example import find_chunk_boundaries
from collections import Counter
chunk_token = b"<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path:str,
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
            pretoken_bytes[tuple(i.encode("utf-8") for i in prestr)] = counts
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
        vocab[len(vocab)] = pair_max[0]+pair_max[1]
        for bytestuple,frequency in pretoken_bytes.items():
            



    return (vocab,merges)



    raise NotImplementedError


if __name__ == "__main__":
    train_bpe("data/TinyStoriesV2-GPT4-valid.txt",10000,["<|endoftext|>"])