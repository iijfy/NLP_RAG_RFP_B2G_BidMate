import torch # GPU 체크를 위해 PyTorch를 불러옵니다.
from langchain_huggingface import HuggingFaceEmbeddings

def get_bge_m3_embeddings() -> HuggingFaceEmbeddings:
    # 1. 디바이스 자동 결정 로직
    # 비유: 작업장에 가서 '전동 드릴(GPU)'이 있는지 확인하고, 없으면 '수동 드라이버(CPU)'를 잡는 것과 같습니다.
    if torch.cuda.is_available():
        device = "cuda"      # NVIDIA GPU가 있는 경우
    elif torch.backends.mps.is_available():
        device = "mps"       # Mac M1/M2/M3 칩셋(Apple Silicon)인 경우
    else:
        device = "cpu"       # 둘 다 없으면 기본 CPU 사용
    
    print(f"현재 임베딩 모델이 사용하는 장치: {device}")

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )