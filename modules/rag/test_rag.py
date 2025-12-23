# modules/rag/test_rag.py
from modules.rag import answer


def main():
    q = "차세대 학사정보시스템 구축 사업의 주요 요구사항은?"
    out = answer(q, k=3)
    print(out)


if __name__ == "__main__":
    main()
