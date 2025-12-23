# modules/retrieval/test_search.py
from modules.retrieval import search


def main():
    docs = search("차세대 학사정보시스템 구축 요구사항", k=3)

    print("\n[OK] search results:")
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] meta={d.metadata}")
        print(d.page_content[:300])


if __name__ == "__main__":
    main()
